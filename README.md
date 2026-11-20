# Self-Supervised Genomic Representation Learning (SSGRL)

Learns embeddings for short DNA sequencing reads **without labels**, using
[SimSiam](https://arxiv.org/abs/2011.10566), then uses them to retrieve similar
reads by nearest-neighbour search.

Each 150 bp read is encoded as a binary *k-mer matrix* (the "Grimm filter"):
one row per position, one column per possible k-mer (k=3 → 148 × 64). Two
randomly augmented views of the same read are pushed together in embedding
space; the stop-gradient and predictor of SimSiam keep the embeddings from
collapsing.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10+
python -m pytest                         # ~15 s

# query reads simulated from the RaTG13 genome (replaces the old missing reads/ folder)
python -m ssgrl.simulate data/RaTG13.fasta -n 500 -o data/sim_RaTG13.fq

# train on the bundled 100k bat reads
python -m ssgrl.train --epochs 10 --batch-size 256 --queries data/sim_RaTG13.fq

# evaluate a saved encoder later
python -m ssgrl.evaluate runs/<run>/backbone.keras --queries data/sim_RaTG13.fq
```

`python -m ssgrl.train -h` lists every option (k, augmentation strengths,
backbone, widths, learning rate, `--mlflow`, ...).

A run directory contains `backbone.keras` (the encoder to use for embeddings),
`simsiam.weights.h5`, `config.json`, `history.json/csv`, `loss.png`,
`metrics.json`, `tsne.png` and, with `--queries`, `neighbours.tsv`.

## How it works

| Piece | File | Notes |
|---|---|---|
| FASTA/FASTQ reading, k-mer encoding | `ssgrl/encoding.py` | Reads stored as one compact uint8 array (cached as `.npy`); k-mer matrices built on the fly |
| Augmentations, `tf.data` pipeline | `ssgrl/data.py` | Random substitutions, reverse complement, random crop/shift |
| Model | `ssgrl/model.py` | `resnet1d` (k-mers as channels, default) or `resnet2d` (matrix as image, the original idea) → attention pooling → SimSiam projector/predictor |
| Training | `ssgrl/train.py` | SGD + cosine decay, best-val checkpoint |
| Evaluation | `ssgrl/evaluate.py` | Retrieval recall@k vs a bag-of-k-mers baseline, neighbour report with edit distance, t-SNE |
| Read simulator | `ssgrl/simulate.py` | Samples reads from a reference with substitution errors |

**Data split** (by position in the file, reproducible):
`[ train | validation (--val-size) | eval database (--eval-size) ]`.

**Retrieval benchmark.** Each eval-database read gets a noisy copy
(substitutions, reverse complement, crop). Recall@k is the fraction of noisy
copies whose nearest neighbour (cosine) among the database is the original.
`kmer_profile_baseline` does the same with plain 3-mer count vectors.

**Collapse monitor.** `z_std` is the per-dimension standard deviation of the
l2-normalised projections × √dim: ≈1 is healthy, ≈0 means every read maps to
the same point.

## Current results (Nov 2026, CPU, 6 epochs, 96k training reads)

Retrieval on 2,000 held-out reads, recall@1 / @5 / @10:

| Encoder | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| 3-mer count profile (baseline) | 0.218 | 0.302 | 0.337 |
| `resnet1d`, no strand invariance | 0.127 | 0.240 | 0.317 |
| `resnet1d`, strand-invariant (default) | 0.201 | **0.331** | **0.405** |

Training is stable (loss → −0.97, `z_std` ≈ 0.84, so no complete collapse),
and the learned encoder beats the baseline at recall@5/10. It does **not** yet
beat it at recall@1. The main reason is *dimensional* collapse: the 256-d
embeddings have an effective rank of only ~5, so many unrelated reads land
close together. The most promising next step is a loss that explicitly
decorrelates dimensions (VICReg or Barlow Twins), followed by longer training
on a GPU.

## Data

- `data/bat2100k.unmapped.fa` – 100,000 unmapped 150 bp reads from a bat sample
- `data/RaTG13.fasta` – bat coronavirus RaTG13 genome (MN996532.1, 29,855 bp)
- `data/RmYN01genome_NMDC60013004-01.fasta` – RmYN02 bat coronavirus genome

## Changes from the 2022 version

The original scripts are kept in `legacy/` for reference. The rewrite fixes:

- **Would not run:** missing `resnet_cifar10_v2` module; missing `reads/` test
  files; requirements were a macOS environment dump (Apple-only TensorFlow,
  unrelated packages) missing several real dependencies; batch size/epochs
  parsed as floats; empty validation slice (`alldbidx[4000:1000]`);
  `KeyError` on a non-existent `reconstruction_loss`; hard-coded 2022 model paths;
  `mlflow.sklearn.log_model` called on a learning-rate schedule.
- **Training bugs:** no stop-gradient on the target branch (SimSiam's key
  ingredient); an extra non-SimSiam loss term; gradients zipped with the wrong
  variable list; a meaningless `Accuracy(p1, p2)` metric; image-style
  `Rescaling(1/127.5, -1)` on 0/1 input; ReLU on the final projection.
- **Augmentation:** the old one rolled the matrix along the k-mer axis, which
  relabels every k-mer. The new ones (substitutions, reverse complement, crop)
  mimic real differences between reads of the same locus.
- **Speed:** vectorised encoding and a `tf.data` pipeline instead of Python loops
  writing one `.npz` per read and eager-mode augmentation.
- **Compatibility:** Keras 3 / TensorFlow 2.16+, Python 3.10–3.13.
