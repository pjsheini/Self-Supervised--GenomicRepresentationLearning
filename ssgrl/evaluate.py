"""Evaluation: retrieval benchmark, nearest-neighbour report, t-SNE.

Retrieval benchmark (replaces the legacy "print the neighbours" loop with a number):
take held-out reads as a database, make a noisy copy of each (substitutions,
reverse complement, crop - the same kinds of change as training), embed both,
and measure how often the original read is the query's nearest neighbour
(recall@k). A bag-of-k-mers profile is reported as a non-learned baseline.

    python -m ssgrl.evaluate runs/<run>/backbone.keras --reads data/bat2100k.unmapped.fa \
        --queries data/sim_RaTG13.fq
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import editdistance
import numpy as np

from .data import AugmentConfig, augment_numpy, make_embed_dataset
from .encoding import decode, kmer_profile, load_reads, read_sequences, reverse_complement


def embed(backbone, reads: np.ndarray, k: int = 3, batch_size: int = 512) -> np.ndarray:
    return np.asarray(backbone.predict(make_embed_dataset(reads, batch_size, k), verbose=0))


def _normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn(db: np.ndarray, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Cosine kNN. Returns (indices, similarities), each (n_queries, k)."""
    sims = _normalize(queries) @ _normalize(db).T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return idx, np.take_along_axis(sims, idx, axis=1)


def recall_at_k(db_emb, q_emb, ks=(1, 5, 10)) -> dict[str, float]:
    idx, _ = knn(db_emb, q_emb, max(ks))
    truth = np.arange(len(q_emb))[:, None]
    return {f"recall@{k}": float((idx[:, :k] == truth).any(axis=1).mean()) for k in ks}


def retrieval_benchmark(backbone, db_reads: np.ndarray, k: int = 3,
                        aug: AugmentConfig | None = None, seed: int = 0) -> dict:
    aug = aug or AugmentConfig()
    queries = augment_numpy(db_reads, aug, seed=seed)
    out = {"n_database": int(len(db_reads)), "augment": asdict(aug)}
    out["kmer_profile_baseline"] = recall_at_k(kmer_profile(db_reads, k), kmer_profile(queries, k))
    if backbone is not None:
        out["learned"] = recall_at_k(embed(backbone, db_reads, k), embed(backbone, queries, k))
    return out


def min_strand_edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    sa = decode(a).strip("N")
    return min(editdistance.eval(sa, decode(b).strip("N")),
               editdistance.eval(sa, decode(reverse_complement(b)).strip("N")))


def neighbour_report(backbone, query_reads, db_reads, query_names=None, k: int = 5,
                     kmer: int = 3) -> list[dict]:
    """For each query, its k nearest database reads with cosine similarity and edit distance."""
    q_emb, db_emb = embed(backbone, query_reads, kmer), embed(backbone, db_reads, kmer)
    idx, sims = knn(db_emb, q_emb, k)
    rows = []
    for qi in range(len(query_reads)):
        for rank, (di, s) in enumerate(zip(idx[qi], sims[qi]), 1):
            rows.append({"query": query_names[qi] if query_names else qi, "rank": rank,
                         "db_index": int(di), "cosine": float(s),
                         "edit_distance": min_strand_edit_distance(query_reads[qi], db_reads[di])})
    return rows


def write_tsv(rows: list[dict], path):
    if not rows:
        return
    with open(path, "w") as fh:
        fh.write("\t".join(rows[0]) + "\n")
        for r in rows:
            fh.write("\t".join(str(v) for v in r.values()) + "\n")


def plot_tsne(emb: np.ndarray, path, labels=None, title="t-SNE of read embeddings"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    xy = TSNE(n_components=2, init="pca", perplexity=min(30, len(emb) - 1), random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6, 6))
    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6)
    else:
        labels = np.asarray(labels)
        for lab in np.unique(labels):
            m = labels == lab
            ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.6, label=str(lab))
        ax.legend(markerscale=4)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def load_query_reads(path, read_len=150):
    names, rows = [], []
    from .encoding import encode
    for name, seq in read_sequences(path):
        r = np.full(read_len, 4, np.uint8)
        c = encode(seq[:read_len]); r[:len(c)] = c
        names.append(name.split()[0]); rows.append(r)
    return np.stack(rows), names


def main(argv=None):
    import keras
    from . import model  # noqa: F401  registers AttentionPooling for loading

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("backbone", help="path to a saved backbone .keras file")
    p.add_argument("--reads", default="data/bat2100k.unmapped.fa")
    p.add_argument("--db-size", type=int, default=2000, help="reads (from the end of the file) used as database")
    p.add_argument("--queries", help="FASTA/FASTQ of query reads for a neighbour report")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", default=None, help="output directory (default: next to the backbone)")
    a = p.parse_args(argv)

    bb = keras.saving.load_model(a.backbone)
    out = Path(a.out or Path(a.backbone).parent); out.mkdir(parents=True, exist_ok=True)
    reads = load_reads(a.reads)
    db = reads[-a.db_size:]
    res = retrieval_benchmark(bb, db, a.k)
    print(json.dumps(res, indent=2))
    (out / "eval_metrics.json").write_text(json.dumps(res, indent=2))
    if a.queries:
        q, names = load_query_reads(a.queries)
        rows = neighbour_report(bb, q, db, names, kmer=a.k)
        write_tsv(rows, out / "neighbours.tsv")
        print(f"wrote {out/'neighbours.tsv'}")


if __name__ == "__main__":
    main()
