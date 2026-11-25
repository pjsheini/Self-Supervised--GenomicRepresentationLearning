"""Compare a trained encoder against reference baselines on read retrieval.

    python -m ssgrl.compare --backbone runs/full_vicreg/backbone.keras

Every method is scored on the same held-out reads, de-duplicated across strands,
with the same noisy queries: recall@1/5/10, plus recall@1 for each kind of change
on its own, plus wall-clock time. Results go to JSON and a markdown table.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from .baselines import (MinHashSketch, canonical_kmer_profile, kmer_profile,  # noqa: E402
                        minimap2_rank, recall_from_ranks)
from .data import AugmentConfig, augment_numpy  # noqa: E402
from .encoding import load_reads  # noqa: E402
from .evaluate import effective_rank, knn, recall_at_k, unique_reads  # noqa: E402

RANK_K = 10


def _embedding_ranks(db_emb, q_emb):
    return knn(db_emb, q_emb, RANK_K)[0]


def build_methods(backbone=None, k: int = 3, skip: tuple[str, ...] = (),
                  dnabert_s: str | None = None):
    """Each method maps (db_reads, query_reads) -> (n_queries, RANK_K) ranked indices."""
    methods: dict[str, callable] = {}

    def add_profile(name, kk, norm):
        methods[name] = lambda db, q, kk=kk, norm=norm: _embedding_ranks(
            kmer_profile(db, kk, norm), kmer_profile(q, kk, norm))

    def add_canonical(name, kk):
        methods[name] = lambda db, q, kk=kk: _embedding_ranks(
            canonical_kmer_profile(db, kk), canonical_kmer_profile(q, kk))

    add_profile(f"{k}-mer profile (l1)", k, "l1")
    add_profile("4-mer profile (l1) / TNF", 4, "l1")
    add_profile("4-mer counts (cosine)", 4, "none")
    add_canonical("canonical 4-mer profile", 4)
    add_canonical("canonical 6-mer profile", 6)
    methods["Mash (k=21, s=1000)"] = lambda db, q: MinHashSketch().rank(db, q, RANK_K)
    methods["minimap2 (preset sr)"] = lambda db, q: minimap2_rank(db, q, RANK_K)

    if dnabert_s:
        from .baselines import DNABERT_S
        model = DNABERT_S(dnabert_s)
        methods["DNABERT-S"] = lambda db, q: _embedding_ranks(model.embed(db), model.embed(q))

    if backbone is not None:
        from .evaluate import embed
        methods["SSGRL (this repo)"] = lambda db, q: _embedding_ranks(
            embed(backbone, db, k), embed(backbone, q, k))

    return {n: f for n, f in methods.items() if n not in skip}


def run(methods, db, aug: AugmentConfig, seed: int = 0, backbone=None, k: int = 3) -> dict:
    """Score every method on the combined queries and on each change alone."""
    single = {"substitutions": AugmentConfig(aug.sub_rate, 0.0, aug.min_keep, 0.0),
              "reverse_complement": AugmentConfig(0.0, 1.0, aug.min_keep, 0.0),
              "crop": AugmentConfig(0.0, 0.0, aug.min_keep, 1.0)}
    queries = {"all": augment_numpy(db, aug, seed=seed)}
    queries.update({n: augment_numpy(db, c, seed=seed) for n, c in single.items()})

    results = {}
    for name, fn in methods.items():
        row: dict = {}
        t0 = time.time()
        ranks = fn(db, queries["all"])
        row.update(recall_from_ranks(ranks))
        row["seconds"] = round(time.time() - t0, 1)
        row["by_augmentation"] = {}
        for qname in single:
            r = fn(db, queries[qname])
            row["by_augmentation"][qname] = recall_from_ranks(r, (1,))["recall@1"]
        results[name] = row
        print(f"  {name:28s} R@1={row['recall@1']:.3f} R@5={row['recall@5']:.3f} "
              f"R@10={row['recall@10']:.3f} {row['seconds']}s", flush=True)

    if backbone is not None:
        from .evaluate import embed
        results["SSGRL (this repo)"]["effective_rank"] = round(
            effective_rank(embed(backbone, db, k)), 1)
    return results


def to_markdown(results: dict, n_db: int) -> str:
    head = (f"Retrieval of {n_db} unique held-out reads from noisy copies.\n\n"
            "| Method | recall@1 | recall@5 | recall@10 | subs only | RC only | crop only | time |\n"
            "|---|---|---|---|---|---|---|---|\n")
    rows = []
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["recall@1"]):
        b = r["by_augmentation"]
        rows.append(f"| {name} | {r['recall@1']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
                    f"{b['substitutions']:.3f} | {b['reverse_complement']:.3f} | {b['crop']:.3f} | "
                    f"{r['seconds']}s |")
    return head + "\n".join(rows) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", help="path to a saved backbone.keras (omit for baselines only)")
    p.add_argument("--reads", default="data/bat2100k.unmapped.fa")
    p.add_argument("--db-size", type=int, default=2000,
                   help="reads taken from the end of the file, before de-duplication")
    p.add_argument("--limit", type=int, default=None, help="cap the de-duplicated database")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--dnabert-s", help="local path or model id for the DNABERT-S baseline")
    p.add_argument("--skip", default="", help="comma-separated method names to skip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/retrieval_comparison")
    a = p.parse_args(argv)

    backbone = None
    if a.backbone:
        import keras

        from . import model  # noqa: F401  registers the custom layers
        backbone = keras.saving.load_model(a.backbone)

    reads = load_reads(a.reads)[-a.db_size:]
    db = unique_reads(reads)
    if a.limit:
        db = db[: a.limit]
    print(f"database: {len(db)} unique reads (of {len(reads)})")

    aug = AugmentConfig()
    methods = build_methods(backbone, a.k, tuple(s.strip() for s in a.skip.split(",") if s.strip()),
                            a.dnabert_s)
    results = run(methods, db, aug, a.seed, backbone, a.k)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_database": len(db), "n_before_dedup": len(reads), "reads": a.reads,
               "k": a.k, "seed": a.seed, "results": results}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(to_markdown(results, len(db)))
    print(f"\n{out.with_suffix('.md')}\n")
    print(to_markdown(results, len(db)))


if __name__ == "__main__":
    main()
