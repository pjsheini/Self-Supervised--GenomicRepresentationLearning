"""Simulate short reads from a reference genome.

Replaces the missing `reads/reads_(RaTG13)_{1,2}.fq` files the legacy code
expected. Reads are sampled uniformly from both strands with a per-base
substitution error rate, and written as FASTQ (constant quality).

    python -m ssgrl.simulate data/RaTG13.fasta -n 2000 -o data/sim_RaTG13.fq
"""
from __future__ import annotations

import argparse

import numpy as np

from .encoding import decode, encode, read_sequences, reverse_complement


def simulate_reads(genome: np.ndarray, n: int, read_len: int = 150, error_rate: float = 0.005,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (reads (n, read_len) uint8, start positions, strand flags)."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(genome) - read_len + 1, n)
    strands = rng.random(n) < 0.5
    reads = np.stack([genome[s:s + read_len] for s in starts]).astype(np.uint8)
    reads[strands] = reverse_complement(reads[strands])
    err = (rng.random(reads.shape) < error_rate) & (reads < 4)
    reads[err] = (reads[err] + rng.integers(1, 4, err.sum())) % 4
    return reads, starts, strands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference")
    p.add_argument("-n", "--num-reads", type=int, default=2000)
    p.add_argument("-l", "--read-len", type=int, default=150)
    p.add_argument("-e", "--error-rate", type=float, default=0.005)
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args(argv)

    name, seq = next(read_sequences(a.reference))
    genome = encode(seq)
    reads, starts, strands = simulate_reads(genome, a.num_reads, a.read_len, a.error_rate, a.seed)
    tag = name.split()[0]
    with open(a.out, "w") as fh:
        for i, (r, s, st) in enumerate(zip(reads, starts, strands)):
            fh.write(f"@{tag}_sim{i} pos={s} strand={'-' if st else '+'}\n{decode(r)}\n+\n{'I' * a.read_len}\n")
    print(f"wrote {len(reads)} reads to {a.out}")


if __name__ == "__main__":
    main()
