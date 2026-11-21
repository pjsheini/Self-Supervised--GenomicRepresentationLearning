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

On 2 CPU cores a VICReg epoch over 96k reads takes about 15 minutes (SimSiam
about 10); a GPU is much faster. For a quick check, add `--max-train 20000
--epochs 2`.

A run directory (`runs/<timestamp>/` unless you pass `--out`) contains:

| File | What it is |
|---|---|
| `backbone.keras` | the trained encoder (read → 256-d embedding); this is the file to use |
| `model.weights.h5` | full model weights from the best validation epoch, including the projection head |
| `config.json` | every setting used for the run |
| `history.json`, `history.csv`, `loss.png` | per-epoch training and validation loss and `z_std` |
| `metrics.json` | retrieval benchmark before and after training |
| `tsne.png` | 2-D t-SNE map of 1,000 held-out read embeddings |
| `neighbours.tsv` | with `--queries`: the 5 nearest database reads of each query, with cosine similarity and edit distance |

### Common options

`python -m ssgrl.train -h` lists all of them.

| Option | Default | Meaning |
|---|---|---|
| `--loss` | `vicreg` | `vicreg`, `barlow` or `simsiam` |
| `--reads` | `data/bat2100k.unmapped.fa` | training reads, FASTA or FASTQ, optionally `.gz` |
| `--read-len` | 150 | reads are truncated or N-padded to this length |
| `--k` | 3 | k-mer size; the matrix has 4^k columns |
| `--max-train` | all | cap on training reads |
| `--epochs`, `--batch-size` | 10, 256 | |
| `--backbone` | `resnet1d` | or `resnet2d`, the original image-style design (slower) |
| `--sub-rate`, `--rc-prob`, `--min-keep`, `--crop-prob` | 0.02, 0.5, 0.7, 0.8 | augmentation strengths |
| `--no-rc-invariant`, `--no-masked-pooling` | off | turn off strand averaging / padding masks |
| `--mlflow` | off | log the run to MLflow (if installed) |

### Using a trained encoder in your own code

```python
import keras
import ssgrl.model  # registers the custom layers needed to load the encoder
from ssgrl.encoding import load_reads
from ssgrl.evaluate import embed, knn

encoder = keras.saving.load_model("runs/<run>/backbone.keras")
reads = load_reads("my_reads.fq")            # (n, 150) array of base codes
emb = embed(encoder, reads)                  # (n, 256) embeddings
idx, sims = knn(emb, emb[:10], k=5)          # 5 nearest reads for the first 10
```

### Your own data

Any FASTA/FASTQ file of short reads works with `--reads`. The last
`--eval-size` + `--val-size` reads (default 2,000 each) are held out, so
shuffle the file first if it is sorted. Reads longer than `--read-len` are
truncated; shorter ones are padded with N, which the model ignores.

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

Among the 500 simulated RaTG13 reads, 3 have a bat read within 30 edits as
their nearest neighbour (best: 9 edits); the rest have no close match (median
78 edits).

## Next steps

- **Train longer on a GPU.** Training loss was still falling after 6 epochs
  (validation loss had levelled off).
- **Close the gap on single changes.** With only substitutions or only a crop
  applied, the encoder is still a little behind the plain 3-mer profile. It
  wins overall because it handles strand flips and combined changes. Try a
  larger VICReg expander (the paper uses 8192; here 1024) and tune the
  augmentation strengths.
- **Follow up the RaTG13 hits.** BLAST the 3 bat reads that sit close to
  RaTG13 reads.
- **Look at the duplicates.** 35% of the held-out bat reads are exact copies,
  possibly rRNA or repeats; filtering them before training may help.

## Data

- `data/bat2100k.unmapped.fa` – 100,000 unmapped 150 bp reads from a bat sample
- `data/RaTG13.fasta` – bat coronavirus RaTG13 genome (MN996532.1, 29,855 bp)
- `data/RmYN01genome_NMDC60013004-01.fasta` – RmYN02 bat coronavirus genome
- `data/sim_RaTG13.fq` – 500 reads simulated from RaTG13 (0.5% errors), used as `--queries`

Loading a read file writes a `.npy` cache next to it (git-ignored), so later
runs start instantly.

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
