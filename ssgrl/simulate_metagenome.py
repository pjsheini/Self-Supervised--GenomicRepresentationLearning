"""Build a CAMI-style synthetic metagenome for the read-binning benchmark.

CAMI2 itself (and DNABERT-S's packaged copy of it) is not reachable from this
workspace, so this writes the same file layout from reference genomes you supply:

    clustering_0.tsv   labelled split, used only to pick the cosine threshold
    binning_5.tsv      the split that is actually clustered and scored
    binning_6.tsv      a second sample, as in DNABERT-S

Each file is `sequence<TAB>label`, exactly DNABERT-S's format, so
`python -m ssgrl.binning --data <dir>` runs unchanged on either source.

Abundances follow a log-normal distribution, as in CAMI's own simulations, so
species are unevenly represented and the task stays hard.

    python -m ssgrl.simulate_metagenome /tmp/genomes/*.fa -o data/metagenome_sim
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .encoding import decode, encode, read_sequences
from .simulate import simulate_reads


def write_split(path: Path, reads: np.ndarray, labels: list[str]) -> None:
    with open(path, "w") as fh:
        fh.write("sequence\tlabel\n")
        for r, lab in zip(reads, labels):
            fh.write(f"{decode(r)}\t{lab}\n")


def sample_split(genomes: dict[str, np.ndarray], n_reads: int, read_len: int,
                 error_rate: float, seed: int) -> tuple[np.ndarray, list[str]]:
    """Draw fragments from each genome in proportion to a log-normal abundance.

    `read_len` is the fragment length: 150 for short reads, or a few thousand to
    stand in for assembled contigs, which is the scale DNABERT-S bins at (it drops
    anything under 2500 bp).
    """
    rng = np.random.default_rng(seed)
    names = sorted(genomes)
    abundance = rng.lognormal(0.0, 1.0, len(names))
    abundance /= abundance.sum()
    counts = np.maximum((abundance * n_reads).astype(int), 20)

    reads, labels = [], []
    for name, count in zip(names, counts):
        g = genomes[name]
        if len(g) < read_len + 1:
            continue
        r, _, _ = simulate_reads(g, int(count), read_len, error_rate,
                                 seed=seed + abs(hash(name)) % 10_000)
        reads.append(r)
        labels.extend([name] * len(r))
    reads = np.concatenate(reads)
    order = rng.permutation(len(reads))
    return reads[order], [labels[i] for i in order]


def load_genomes(paths: list[str]) -> dict[str, np.ndarray]:
    genomes = {}
    for p in paths:
        name = Path(p).stem.replace(".fa", "").replace(".fasta", "")
        seqs = [encode(s) for _, s in read_sequences(p)]
        if seqs:
            genomes[name] = np.concatenate(seqs)
    return genomes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("genomes", nargs="+", help="reference FASTA files, one species each")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("-n", "--reads-per-split", type=int, default=4000)
    p.add_argument("-l", "--read-len", type=int, default=150,
                   help="fragment length; use >=2500 for contig-scale binning")
    p.add_argument("-e", "--error-rate", type=float, default=0.005)
    p.add_argument("-s", "--seed", type=int, default=0)
    a = p.parse_args(argv)

    genomes = load_genomes(a.genomes)
    if len(genomes) < 2:
        raise SystemExit("need at least two genomes")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, seed in [("clustering_0", a.seed), ("binning_5", a.seed + 1),
                        ("binning_6", a.seed + 2)]:
        reads, labels = sample_split(genomes, a.reads_per_split, a.read_len, a.error_rate, seed)
        write_split(out / f"{split}.tsv", reads, labels)
        from collections import Counter
        print(f"{split}: {len(reads)} reads, {len(set(labels))} species, "
              f"largest {Counter(labels).most_common(1)[0][1]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
