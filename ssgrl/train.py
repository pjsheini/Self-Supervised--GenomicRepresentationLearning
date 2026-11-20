"""Train a SimSiam encoder on short reads.

    python -m ssgrl.train --reads data/bat2100k.unmapped.fa --epochs 10 --batch-size 256

Data split (by position in the file, so it is reproducible and never overlaps):
    [ train ... | validation (--val-size) | eval database (--eval-size) ]
The eval database is the same slice `python -m ssgrl.evaluate` uses by default.

Outputs in --out (default runs/<timestamp>/): backbone.keras, simsiam.weights.h5,
config.json, history.json, loss.png, metrics.json, tsne.png, and neighbours.tsv
if --queries is given.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from .data import AugmentConfig, make_ssl_dataset  # noqa: E402
from .encoding import load_reads  # noqa: E402
from .evaluate import (embed, load_query_reads, neighbour_report, plot_tsne,  # noqa: E402
                       retrieval_benchmark, write_tsv)
from .model import build_simsiam  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--reads", default="data/bat2100k.unmapped.fa")
    d.add_argument("--read-len", type=int, default=150)
    d.add_argument("--k", type=int, default=3, help="k-mer size (matrix has 4**k columns)")
    d.add_argument("--max-train", type=int, default=None, help="cap on training reads")
    d.add_argument("--val-size", type=int, default=2000)
    d.add_argument("--eval-size", type=int, default=2000)
    d.add_argument("--queries", help="FASTA/FASTQ of query reads (e.g. simulated RaTG13) for a neighbour report")
    a = p.add_argument_group("augmentation")
    a.add_argument("--sub-rate", type=float, default=0.02)
    a.add_argument("--rc-prob", type=float, default=0.5)
    a.add_argument("--min-keep", type=float, default=0.7)
    a.add_argument("--crop-prob", type=float, default=0.8)
    m = p.add_argument_group("model")
    m.add_argument("--backbone", choices=["resnet1d", "resnet2d"], default="resnet1d")
    m.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    m.add_argument("--blocks", type=int, default=2, help="residual blocks per stage")
    m.add_argument("--no-rc-invariant", dest="rc_invariant", action="store_false",
                   help="don't average embeddings over both strands")
    m.add_argument("--proj-dim", type=int, default=256)
    m.add_argument("--pred-hidden", type=int, default=64)
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=None, help="default: 0.05 * batch_size / 256 (SimSiam rule)")
    t.add_argument("--weight-decay", type=float, default=5e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None)
    t.add_argument("--mlflow", action="store_true", help="log params/metrics to MLflow if installed")
    return p.parse_args(argv)


def split_reads(reads, val_size, eval_size, max_train=None):
    n = len(reads)
    if n <= val_size + eval_size:
        raise SystemExit(f"only {n} reads; need more than val_size+eval_size={val_size + eval_size}")
    eval_db = reads[n - eval_size:]
    val = reads[n - eval_size - val_size: n - eval_size]
    train = reads[: n - eval_size - val_size]
    if max_train:
        train = train[:max_train]
    return train, val, eval_db


def save_loss_plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, ax in zip(["loss", "z_std"], axes):
        ax.plot(history[key], label="train")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label="validation")
        ax.set_xlabel("epoch"); ax.set_title(key); ax.legend()
    axes[0].axhline(-1, ls=":", c="grey")
    axes[1].set_ylim(0, 1.1)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    keras.utils.set_random_seed(args.seed)
    out = Path(args.out or f"runs/{datetime.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    reads = load_reads(args.reads, args.read_len)
    train, val, eval_db = split_reads(reads, args.val_size, args.eval_size, args.max_train)
    print(f"reads: train={len(train)} val={len(val)} eval_db={len(eval_db)}")

    aug = AugmentConfig(args.sub_rate, args.rc_prob, args.min_keep, args.crop_prob)
    train_ds = make_ssl_dataset(train, args.batch_size, args.k, aug, seed=args.seed)
    val_ds = make_ssl_dataset(val, args.batch_size, args.k, aug, shuffle=False)

    input_shape = (args.read_len - args.k + 1, 4 ** args.k)
    model = build_simsiam(input_shape, args.backbone, tuple(args.widths), args.blocks,
                          args.proj_dim, args.pred_hidden, args.weight_decay, args.rc_invariant)
    model.backbone.summary(print_fn=lambda s, **_: None)

    lr = args.lr or 0.05 * args.batch_size / 256
    steps = max(1, args.epochs * (len(train) // args.batch_size))
    schedule = keras.optimizers.schedules.CosineDecay(lr, decay_steps=steps)
    model.compile(optimizer=keras.optimizers.SGD(schedule, momentum=0.9))

    metrics = {"before_training": retrieval_benchmark(model.backbone, eval_db, args.k, aug)}
    print("before training:", metrics["before_training"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out / "simsiam.weights.h5"), monitor="val_loss",
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
    mlflow = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.start_run()
            mlflow.log_params(vars(args))
        except ImportError:
            print("mlflow not installed; skipping")
            mlflow = None

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks,
                     shuffle=False, verbose=2)  # dataset is already shuffled
    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    (out / "history.json").write_text(json.dumps(history, indent=2))
    save_loss_plot(history, out / "loss.png")

    model.load_weights(out / "simsiam.weights.h5")  # best validation epoch
    model.backbone.save(out / "backbone.keras")

    metrics["after_training"] = retrieval_benchmark(model.backbone, eval_db, args.k, aug)
    metrics["final"] = {k: v[-1] for k, v in history.items()}
    print("after training:", metrics["after_training"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    sample = eval_db[np.random.default_rng(0).choice(len(eval_db), min(1000, len(eval_db)), replace=False)]
    plot_tsne(embed(model.backbone, sample, args.k), out / "tsne.png")

    if args.queries:
        q, names = load_query_reads(args.queries, args.read_len)
        write_tsv(neighbour_report(model.backbone, q, eval_db, names, kmer=args.k), out / "neighbours.tsv")

    if mlflow:
        for k, v in metrics["after_training"]["learned"].items():
            mlflow.log_metric(k.replace("@", "_at_"), v)
        mlflow.log_artifacts(str(out))
        mlflow.end_run()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
