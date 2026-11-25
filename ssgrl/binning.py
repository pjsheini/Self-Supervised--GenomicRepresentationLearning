"""Read-level binning benchmark: group reads by the genome they came from.

Protocol follows DNABERT-S's `eval_binning.py` (Zhou et al., Bioinformatics 2025)
so numbers are comparable with that paper and with Çelikkanat et al. (2024):

  1. embed every sequence and l2-normalise,
  2. pick a cosine threshold from a labelled *clustering* split - the 30th
     percentile of each sequence's similarity to its own class centre,
  3. cluster the *binning* split with their greedy KMedoid, dropping bins
     smaller than `min_bin_size`,
  4. match clusters to species with the Hungarian algorithm, then count bins
     above each F1 threshold. "High-quality" means F1 > 0.9.

Data. CAMI2 (the standard source, via DNABERT-S's own release) is not reachable
from this workspace, so `python -m ssgrl.binning.simulate` builds a CAMI-style
synthetic metagenome from reference genomes instead: reads drawn per genome at
set abundances, with sequencing error. `--data` accepts DNABERT-S's own
`binning_{5,6}.tsv` / `clustering_0.tsv` files unchanged, so the same command
runs on CAMI2 wherever those files are available.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

F1_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def class_center_similarity_threshold(emb: np.ndarray, labels: np.ndarray,
                                     percentile: int = 30) -> float:
    """DNABERT-S's threshold: percentile of similarity to one's own class centre.

    Their code takes `percentile_values[-3]`, i.e. the 70th entry of a list built
    at 10..90, which is the 30th percentile counting from the bottom.
    """
    sims = np.zeros(len(emb))
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        center = emb[idx].mean(axis=0)
        sims[idx] = emb[idx] @ center
    sims.sort()
    return float(sims[int(percentile / 100 * len(emb))])


def kmedoid(features: np.ndarray, min_similarity: float, min_bin_size: int = 10,
            max_iter: int = 1000) -> np.ndarray:
    """Greedy cosine KMedoid, a reimplementation of DNABERT-S's `KMedoid`.

    Repeatedly takes the densest unassigned point, refines its medoid three times,
    and claims everything within `min_similarity`. Unassigned reads keep label -1.
    """
    features = features.astype(np.float32)
    sims = features @ features.T
    sims[sims < min_similarity] = 0
    row_sum = sims.sum(axis=1)
    labels = np.full(len(features), -1, dtype=int)

    count = 0
    while np.any(labels == -1) and count <= max_iter:
        count += 1
        i = int(np.argmax(row_sum))
        row_sum[i] = -100
        medoid = features[i]
        available = labels == -1
        idx = np.array([], dtype=int)
        for _ in range(3):
            within = (features @ medoid) >= min_similarity
            idx = np.where(within & available)[0]
            if not len(idx):
                break
            medoid = features[idx].mean(axis=0)
        if not len(idx):
            continue
        labels[idx] = count
        row_sum -= sims[:, idx].sum(axis=1)
        row_sum[idx] = -100

    for lab, n in Counter(labels.tolist()).items():
        if lab != -1 and n < min_bin_size:
            labels[labels == lab] = -1
    return labels


def bin_quality(true_labels: np.ndarray, predicted: np.ndarray) -> dict:
    """Hungarian-matched per-bin F1 and recall, counted above each threshold."""
    import sklearn.metrics
    from scipy.optimize import linear_sum_assignment

    keep = predicted != -1
    true_labels, predicted = true_labels[keep], predicted[keep]
    if not len(true_labels):
        return {"n_bins": 0, "n_assigned": 0,
                "f1_bins": {str(t): 0 for t in F1_THRESHOLDS},
                "recall_bins": {str(t): 0 for t in F1_THRESHOLDS}}

    size = max(true_labels.max(), predicted.max()) + 1
    confusion = np.zeros((size, size), dtype=int)
    for t, p in zip(true_labels, predicted):
        confusion[t, p] += 1
    rows, cols = linear_sum_assignment(confusion, maximize=True)
    mapping = {int(p): int(t) for t, p in zip(rows, cols)}
    aligned = np.array([mapping.get(int(p), -1) for p in predicted])

    f1 = np.sort(sklearn.metrics.f1_score(true_labels, aligned, average=None, zero_division=0))
    rec = np.sort(sklearn.metrics.recall_score(true_labels, aligned, average=None, zero_division=0))
    return {
        "n_bins": int(len(np.unique(predicted))),
        "n_assigned": int(len(true_labels)),
        "assigned_fraction": round(float(keep.mean()), 3),
        "ari": round(float(sklearn.metrics.adjusted_rand_score(true_labels, predicted)), 3),
        "f1_bins": {str(t): int((f1 > t).sum()) for t in F1_THRESHOLDS},
        "recall_bins": {str(t): int((rec > t).sum()) for t in F1_THRESHOLDS},
    }


def read_tsv(path) -> tuple[list[str], np.ndarray]:
    """DNABERT-S format: a header row, then sequence<TAB>label."""
    with open(path) as fh:
        rows = list(csv.reader(fh, delimiter="\t"))[1:]
    return [r[0] for r in rows], np.array([r[1] for r in rows])


def encode_sequences(seqs: list[str], read_len: int) -> np.ndarray:
    """Pad or truncate every sequence to `read_len` base codes."""
    from .encoding import encode
    out = np.full((len(seqs), read_len), 4, dtype=np.uint8)
    for i, s in enumerate(seqs):
        c = encode(s[:read_len])
        out[i, : len(c)] = c
    return out


def tile_sequences(seqs: list[str], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut each sequence into non-overlapping `window`-long tiles.

    Returns the tiles and, for each tile, the index of the sequence it came from,
    so a fixed-length encoder can embed a long contig by averaging over its tiles.
    The trailing partial tile is kept and N-padded unless it is under half a window.
    """
    from .encoding import encode
    tiles, owner = [], []
    for i, s in enumerate(seqs):
        c = encode(s)
        n = max(1, len(c) // window)
        for j in range(n):
            chunk = c[j * window:(j + 1) * window]
            if len(chunk) < window // 2 and j > 0:
                break
            tile = np.full(window, 4, dtype=np.uint8)
            tile[: len(chunk)] = chunk
            tiles.append(tile)
            owner.append(i)
    return np.stack(tiles), np.array(owner)


def pooled_embedding(fn, tiles: np.ndarray, owner: np.ndarray, n_seqs: int) -> np.ndarray:
    """Mean-pool tile embeddings back to one vector per sequence."""
    emb = fn(tiles)
    out = np.zeros((n_seqs, emb.shape[1]), dtype=np.float64)
    counts = np.bincount(owner, minlength=n_seqs).astype(np.float64)
    np.add.at(out, owner, emb)
    return (out / np.maximum(counts, 1)[:, None]).astype(np.float32)


def embedders(backbone=None, k: int = 3, dnabert_s: str | None = None,
              window: int = 150) -> dict[str, tuple]:
    """name -> (embed_fn, mode).

    mode "full" sees each sequence at its full length; "tiled" sees fixed-length
    tiles whose embeddings are averaged per sequence. k-mer profiles are
    length-agnostic so they run on the full sequence, the way TNF binners do.
    SSGRL has a fixed 150 bp input, so it is tiled.
    """
    from .baselines import canonical_kmer_profile, kmer_profile

    fns: dict[str, tuple] = {
        f"{k}-mer profile (l1)": (lambda r: kmer_profile(r, k, "l1"), "full"),
        "4-mer profile (l1) / TNF": (lambda r: kmer_profile(r, 4, "l1"), "full"),
        "canonical 4-mer profile": (lambda r: canonical_kmer_profile(r, 4), "full"),
        "canonical 6-mer profile": (lambda r: canonical_kmer_profile(r, 6), "full"),
    }
    if dnabert_s:
        from .baselines import DNABERT_S
        model = DNABERT_S(dnabert_s)
        fns["DNABERT-S"] = (model.embed, "full")
    if backbone is not None:
        from .evaluate import embed
        fns["SSGRL (this repo)"] = (lambda r: embed(backbone, r, k), "tiled")
        fns[f"canonical 6-mer, {window}bp tiles"] = (
            lambda r: canonical_kmer_profile(r, 6), "tiled")
    return fns


class Split:
    """Holds one split in both representations, so each method gets the one it needs."""

    def __init__(self, seqs: list[str], labels: np.ndarray, window: int):
        self.labels = labels
        self.n = len(seqs)
        max_len = max(len(s) for s in seqs)
        self.full = encode_sequences(seqs, max_len)
        self.tiles, self.owner = tile_sequences(seqs, window)

    def embed(self, fn, mode: str) -> np.ndarray:
        if mode == "full":
            return fn(self.full)
        return pooled_embedding(fn, self.tiles, self.owner, self.n)


def evaluate(fns: dict, clustering: "Split", binning: "Split",
             min_bin_size: int = 10, percentile: int = 30) -> dict:
    """clustering and binning are `Split` objects."""
    results = {}
    for name, (fn, mode) in fns.items():
        t0 = time.time()
        threshold = class_center_similarity_threshold(
            normalize(clustering.embed(fn, mode)), clustering.labels, percentile)
        labels = kmedoid(normalize(binning.embed(fn, mode)), threshold, min_bin_size)
        row = bin_quality(binning.labels, labels)
        row["threshold"] = round(threshold, 4)
        row["seconds"] = round(time.time() - t0, 1)
        results[name] = row
        print(f"  {name:28s} HQ(F1>0.9)={row['f1_bins']['0.9']:3d}  bins={row['n_bins']:3d}  "
              f"ARI={row.get('ari')}  assigned={row.get('assigned_fraction')}  {row['seconds']}s",
              flush=True)
    return results


def to_markdown(results: dict, n_species: int, n_reads: int) -> str:
    head = (f"Read binning: {n_reads} reads from {n_species} genomes, "
            "DNABERT-S protocol (threshold from a labelled split, greedy KMedoid, "
            "Hungarian matching).\n\n"
            "| Method | bins with F1>0.9 | F1>0.7 | F1>0.5 | bins found | ARI | reads binned | time |\n"
            "|---|---|---|---|---|---|---|---|\n")
    rows = []
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["f1_bins"]["0.9"]):
        rows.append(f"| {name} | {r['f1_bins']['0.9']} | {r['f1_bins']['0.7']} | "
                    f"{r['f1_bins']['0.5']} | {r['n_bins']} | {r.get('ari', '-')} | "
                    f"{r.get('assigned_fraction', '-')} | {r['seconds']}s |")
    return head + "\n".join(rows) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="directory with clustering_0.tsv and binning_<sample>.tsv "
                        "(DNABERT-S layout; `ssgrl.simulate_metagenome` writes the same)")
    p.add_argument("--sample", default="5")
    p.add_argument("--backbone", help="path to a saved backbone.keras")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--window", type=int, default=150,
                   help="tile length for fixed-input encoders such as SSGRL")
    p.add_argument("--min-bin-size", type=int, default=10)
    p.add_argument("--percentile", type=int, default=30)
    p.add_argument("--dnabert-s", help="local path or model id for the DNABERT-S baseline")
    p.add_argument("--out", default="results/binning")
    a = p.parse_args(argv)

    backbone = None
    if a.backbone:
        import keras

        from . import model  # noqa: F401
        backbone = keras.saving.load_model(a.backbone)

    data = Path(a.data)
    clu_seqs, clu_labels = read_tsv(data / "clustering_0.tsv")
    bin_seqs, bin_labels = read_tsv(data / f"binning_{a.sample}.tsv")
    counts = Counter(bin_labels.tolist())
    keep = [i for i, l in enumerate(bin_labels) if counts[l] >= a.min_bin_size]
    bin_seqs = [bin_seqs[i] for i in keep]
    bin_labels = bin_labels[keep]

    clu_ids = {l: i for i, l in enumerate(sorted(set(clu_labels)))}
    bin_ids = {l: i for i, l in enumerate(sorted(set(bin_labels)))}
    print(f"clustering split: {len(clu_seqs)} reads, {len(clu_ids)} species")
    print(f"binning split:    {len(bin_seqs)} reads, {len(bin_ids)} species")

    clustering = Split(clu_seqs, np.array([clu_ids[l] for l in clu_labels]), a.window)
    binning = Split(bin_seqs, np.array([bin_ids[l] for l in bin_labels]), a.window)
    print(f"sequence length: {binning.full.shape[1]} bp max, "
          f"{len(binning.tiles)} tiles of {a.window} bp")

    fns = embedders(backbone, a.k, a.dnabert_s, a.window)
    results = evaluate(fns, clustering, binning, a.min_bin_size, a.percentile)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"data": str(data), "sample": a.sample, "n_species": len(bin_ids),
         "n_reads": len(bin_seqs), "results": results}, indent=2))
    md = to_markdown(results, len(bin_ids), len(bin_seqs))
    out.with_suffix(".md").write_text(md)
    print(f"\n{out.with_suffix('.md')}\n\n{md}")


if __name__ == "__main__":
    main()
