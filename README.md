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

# compare against minimap2, Mash and canonical k-mer profiles
python -m ssgrl.compare --backbone runs/<run>/backbone.keras

# read-binning benchmark (DNABERT-S protocol) on a synthetic metagenome
python -m ssgrl.simulate_metagenome data/genomes/*.fa -o data/metagenome_sim -n 1200 -l 5000
python -m ssgrl.binning --data data/metagenome_sim --sample 5 --backbone runs/<run>/backbone.keras
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
| Baselines | `ssgrl/baselines.py` | k-mer profiles, canonical (strand-folded) profiles, Mash MinHash, minimap2 via `mappy`, DNABERT-S |
| Retrieval comparison | `ssgrl/compare.py` | Scores every method on the same queries |
| Binning benchmark | `ssgrl/binning.py` | DNABERT-S protocol: threshold, KMedoid, Hungarian matching, F1 bins |
| Metagenome simulator | `ssgrl/simulate_metagenome.py` | CAMI-style synthetic metagenome in DNABERT-S's TSV layout |

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

**The headline: SSGRL does not beat standard baselines.** Against a plain
3-mer profile it looks good, but that baseline cannot handle both DNA strands.
Counting k-mers *canonically* (folding each k-mer together with its reverse
complement, as Mash and Kraken do) removes that handicap in one line of code
and beats the learned encoder outright, as do Mash and minimap2.

### Read retrieval

1,305 unique held-out reads; each gets a noisy copy (2% substitutions, strand
flip, crop to ≥70%) and we ask whether its nearest neighbour is the original.
Reproduce with `python -m ssgrl.compare --backbone runs/<run>/backbone.keras`.

| Method | recall@1 | recall@5 | recall@10 | subs only | RC only | crop only | time |
|---|---|---|---|---|---|---|---|
| canonical 6-mer profile | **0.730** | **0.852** | **0.880** | **0.935** | 0.997 | 0.706 | 0.4s |
| minimap2 (preset `sr`) | 0.726 | 0.805 | 0.813 | 0.868 | 0.971 | **0.717** | 0.2s |
| Mash (k=21, s=1000) | 0.695 | 0.836 | 0.874 | 0.835 | **1.000** | 0.723 | 5.2s |
| canonical 4-mer profile | 0.663 | 0.808 | 0.844 | 0.878 | 0.995 | 0.659 | 0.1s |
| SSGRL VICReg | 0.569 | 0.753 | 0.799 | 0.752 | **1.000** | 0.597 | 3.3s |
| 4-mer profile / TNF | 0.373 | 0.465 | 0.507 | 0.918 | 0.031 | 0.682 | 0.0s |
| 3-mer profile | 0.344 | 0.448 | 0.477 | 0.870 | 0.033 | 0.634 | 0.1s |
| SSGRL SimSiam | 0.314 | 0.463 | 0.551 | 0.714 | 1.000 | 0.295 | – |

The `RC only` column explains the earlier result: plain profiles score 0.03 on
reverse-complemented queries, so SSGRL's whole advantage over them came from
strand handling, not from anything learned. Canonical counting supplies that
for free and is better on substitutions too.

### Read binning (DNABERT-S protocol)

Threshold from a labelled split, greedy KMedoid clustering, Hungarian matching,
then bins counted by F1 — the procedure in DNABERT-S's `eval_binning.py`, so the
numbers are comparable with that paper. CAMI2 is not reachable from this
workspace, so these runs use a synthetic metagenome built from 6 reference
genomes (`ssgrl.simulate_metagenome`); the same command runs on CAMI2 TSVs
unchanged. Averaged over the two samples:

| Method | bins F1>0.7 | bins F1>0.5 | ARI | time |
|---|---|---|---|---|
| canonical 4-mer profile | **3.5** | **6.0** | **0.509** | 0.8s |
| canonical 6-mer profile | 3.0 | 5.5 | 0.463 | 1.4s |
| 4-mer profile / TNF | 1.5 | 3.5 | 0.313 | 1.0s |
| 3-mer profile | 1.5 | 3.0 | 0.276 | 0.9s |
| SSGRL VICReg | 1.5 | 3.0 | 0.314 | 81s |

A tiled canonical 6-mer control (same 150 bp windows and mean pooling SSGRL
needs for long sequences) scores ARI 0.45, so tiling is not what holds SSGRL
back — the embedding itself is.

**At 150 bp, composition binning does not work at all:** every method lands at
ARI 0.06–0.11 with no bin above F1 0.5. DNABERT-S filters to sequences ≥2500 bp
for exactly this reason. Read-level binning is only meaningful at contig scale,
which an encoder with a fixed 150 bp input can reach only by tiling.

### Training behaviour

| Loss | recall@1 | effective rank (of 256) |
|---|---|---|
| SimSiam | 0.314 | 4.7 |
| Barlow Twins | ~0.41 (20k reads) | 12.5 |
| VICReg | 0.569 | 37.3 |

SimSiam collapsed onto ~5 of 256 dimensions. VICReg's variance and covariance
terms fixed that, which is why it is the default. Small-scale comparison (20k
reads, 5 epochs, recall@1): SimSiam 0.307, Barlow Twins 0.408, VICReg 0.503,
VICReg + masked pooling 0.515.

Among 500 simulated RaTG13 reads, 3 have a bat read within 30 edits as their
nearest neighbour (best: 9 edits); the rest have no close match (median 78).

## Next steps

The benchmarks above say where the real problem is. Ranked by what would change
the conclusion:

1. **Beat canonical k-mer counting, or stop.** That is the honest bar: 0.730
   recall@1 and ARI 0.509, with no training and in under a second. If a learned
   encoder cannot pass it, the learning is not earning its cost.
2. **Fix crop robustness.** SSGRL scores 0.597 on cropped queries against 0.717
   for minimap2. Attention pooling over 150 bp may be discarding positional
   detail; try several pooled scales, or local alignment-style matching.
3. **Give the encoder canonical k-mer counts as input,** rather than making it
   learn strand invariance from scratch. It is already strand-invariant by
   construction, so the input could be too.
4. **Train longer on a GPU** with a larger VICReg expander (the paper uses 8192;
   here 1024). An epoch is ~15 min on 2 CPU cores.
5. **Run real CAMI2** once network access allows (`--data` already accepts the
   DNABERT-S TSV layout), and add DNABERT-S itself with `--dnabert-s`
   (implemented, untested here: Hugging Face is blocked in this workspace).
6. **Support contig-length input natively,** since that is the scale at which
   binning works.
7. **Follow up the RaTG13 hits** by BLASTing those 3 bat reads.

## Data

- `data/bat2100k.unmapped.fa` – 100,000 unmapped 150 bp reads from a bat sample
- `data/RaTG13.fasta` – bat coronavirus RaTG13 genome (MN996532.1, 29,855 bp)
- `data/RmYN01genome_NMDC60013004-01.fasta` – RmYN02 bat coronavirus genome
- `data/sim_RaTG13.fq` – 500 reads simulated from RaTG13 (0.5% errors), used as `--queries`
- `data/genomes/` – 6 reference genomes (from [nf-core/test-datasets](https://github.com/nf-core/test-datasets)) for the synthetic metagenome

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
