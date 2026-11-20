# Self-Supervised Genomic Representation Learning (SSGRL)

Learns embeddings for short DNA sequencing reads **without labels**, using
[VICReg](https://arxiv.org/abs/2105.04906) (default),
[Barlow Twins](https://arxiv.org/abs/2103.03230) or
[SimSiam](https://arxiv.org/abs/2011.10566), then uses them to retrieve similar
reads by nearest-neighbour search.

Each 150 bp read is encoded as a binary *k-mer matrix* (the "Grimm filter"):
one row per position, one column per possible k-mer (k=3 → 148 × 64). Two
randomly augmented views of the same read are pushed together in embedding
space, while VICReg's variance and covariance terms keep the embedding
dimensions spread out and decorrelated (SimSiam relies on a stop-gradient
instead, and on this data it used only ~5 of 256 dimensions).

## Quick start

```bash
pip install -r requirements.txt          # Python 3.10+
python -m pytest                         # ~1 min

# query reads simulated from the RaTG13 genome (replaces the old missing reads/ folder)
python -m ssgrl.simulate data/RaTG13.fasta -n 500 -o data/sim_RaTG13.fq

# train on the bundled 100k bat reads (VICReg by default; --loss barlow|simsiam)
python -m ssgrl.train --epochs 6 --batch-size 256 --queries data/sim_RaTG13.fq

# evaluate a saved encoder later
python -m ssgrl.evaluate runs/<run>/backbone.keras --queries data/sim_RaTG13.fq
```

`python -m ssgrl.train -h` lists every option (loss, k, augmentation strengths,
backbone, widths, optimizer, learning rate, `--mlflow`, ...). On 2 CPU cores an
epoch over 96k reads takes about 11 minutes; a GPU is much faster.

A run directory contains `backbone.keras` (the encoder to use for embeddings),
`simsiam.weights.h5`, `config.json`, `history.json/csv`, `loss.png`,
`metrics.json`, `tsne.png` and, with `--queries`, `neighbours.tsv`.

## How it works

| Piece | File | Notes |
|---|---|---|
| FASTA/FASTQ reading, k-mer encoding | `ssgrl/encoding.py` | Reads stored as one compact uint8 array (cached as `.npy`); k-mer matrices built on the fly |
| Augmentations, `tf.data` pipeline | `ssgrl/data.py` | Random substitutions, reverse complement, random crop/shift |
| Model | `ssgrl/model.py` | `resnet1d` (k-mers as channels, default) or `resnet2d` (matrix as image, the original idea) → attention pooling that ignores N padding → averaged over both strands → VICReg/Barlow expander or SimSiam projector + predictor |
| Training | `ssgrl/train.py` | AdamW with warmup + cosine decay (SGD for SimSiam), best-val checkpoint |
| Evaluation | `ssgrl/evaluate.py` | Retrieval recall@k vs a bag-of-k-mers baseline, neighbour report with edit distance, t-SNE |
| Read simulator | `ssgrl/simulate.py` | Samples reads from a reference with substitution errors |

**Data split** (by position in the file, reproducible):
`[ train | validation (--val-size) | eval database (--eval-size) ]`.

**Retrieval benchmark.** The eval database is first de-duplicated (a read and
its reverse complement count as the same): 35% of the 2,000 held-out bat reads
are exact copies, leaving 1,305 unique reads. Each gets a noisy copy
(substitutions, reverse complement, crop). Recall@k is the fraction of noisy
copies whose nearest neighbour (cosine) among the database is the original.
`kmer_profile_baseline` does the same with plain 3-mer count vectors.
`metrics.json` also reports recall@1 for each kind of change separately and
the embeddings' *effective rank* (how many dimensions are really used).

**Collapse monitor.** `z_std` is the per-dimension standard deviation of the
l2-normalised projections × √dim: ≈1 is healthy, ≈0 means every read maps to
the same point.

## Current results

CPU, 96k training reads, 6 epochs, strand-invariant `resnet1d`. Retrieval
against 1,305 unique held-out reads (see *Retrieval benchmark* above):

| Encoder | recall@1 | recall@5 | recall@10 | effective rank (of 256) |
|---|---|---|---|---|
| 3-mer count profile (baseline) | 0.344 | 0.448 | 0.477 | – |
| SimSiam | 0.314 | 0.463 | 0.551 | 4.7 |
| **VICReg (default)** | **0.569** | **0.753** | **0.799** | **37.3** |

Recall@1 when only one kind of change is applied:

| Change | baseline | VICReg |
|---|---|---|
| substitutions (2%) | 0.870 | 0.752 |
| reverse complement | 0.033 | 1.000 |
| crop to ≥70% + shift | 0.634 | 0.597 |

Loss comparison at small scale (20k reads, 5 epochs; recall@1 / @5 / @10):
SimSiam 0.307 / 0.382 / 0.442, Barlow Twins 0.408 / 0.615 / 0.704,
VICReg 0.503 / 0.690 / 0.752, VICReg + masked pooling 0.515 / 0.696 / 0.766.

**Still open.** On substitutions or crops alone, the learned encoder is
still a little behind the plain 3-mer profile; it wins overall because it
also handles strand and combined changes. Likely next gains: longer training
on a GPU, a larger expander (VICReg papers use 8192), and tuning the
crop/substitution strengths. Among the 500 simulated RaTG13 reads, 3 have a
bat read within 30 edits as nearest neighbour (best: 9 edits); the rest have
no close match (median 78 edits).

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
- **Loss:** VICReg replaces SimSiam as the default; SimSiam suffered
  dimensional collapse on this data and did not beat the 3-mer baseline.
