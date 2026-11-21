"""DNA reading and k-mer ("Grimm filter") encoding.

Reads are stored compactly as uint8 base codes (A=0, C=1, G=2, T=3, N/other=4).
The k-mer one-hot matrix the model sees (L-k+1 rows x 4**k columns) is built on
the fly, so we never write one file per read as the original code did.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

import numpy as np

BASES = "ACGT"
N_CODE = 4
_LUT = np.full(256, N_CODE, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i
_DECODE = np.frombuffer(b"ACGTN", dtype=np.uint8)


def _open(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r", encoding="utf-8", errors="ignore")


def read_sequences(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) from a FASTA or FASTQ file (optionally gzipped)."""
    with _open(path) as fh:
        first = fh.read(1)
        if not first:
            return
        fh.seek(0)
        if first == "@":  # FASTQ: 4 lines per record
            while True:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline().strip()
                fh.readline()
                fh.readline()
                yield header[1:].strip(), seq
        else:  # FASTA, possibly multi-line
            name, chunks = None, []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        yield name, "".join(chunks)
                    name, chunks = line[1:], []
                else:
                    chunks.append(line)
            if name is not None:
                yield name, "".join(chunks)


def encode(seq: str) -> np.ndarray:
    """String -> uint8 base codes."""
    return _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    return _DECODE[np.asarray(codes, dtype=np.uint8)].tobytes().decode()


def reverse_complement(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    rc = np.where(codes < N_CODE, 3 - codes, N_CODE)
    return rc[..., ::-1].astype(np.uint8)


def load_reads(path: str | Path, read_len: int = 150, max_reads: int | None = None,
               cache: bool = True) -> np.ndarray:
    """Load reads into an (n, read_len) uint8 array, N-padding/truncating to read_len.

    A compressed .npy cache is written next to the input so later runs are instant.
    """
    path = Path(path)
    cache_path = path.with_suffix(path.suffix + f".L{read_len}.npy")
    if cache and cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        arr = np.load(cache_path)
        return arr[:max_reads] if max_reads else arr

    rows = []
    for _, seq in read_sequences(path):
        row = np.full(read_len, N_CODE, dtype=np.uint8)
        c = encode(seq[:read_len])
        row[: len(c)] = c
        rows.append(row)
    arr = np.stack(rows) if rows else np.zeros((0, read_len), np.uint8)
    if cache:
        try:
            np.save(cache_path, arr)
        except OSError:
            pass
    return arr[:max_reads] if max_reads else arr


def kmer_indices(codes: np.ndarray, k: int = 3) -> np.ndarray:
    """(..., L) base codes -> (..., L-k+1) k-mer indices in [0, 4**k), -1 if the k-mer has an N."""
    codes = np.asarray(codes).astype(np.int64)
    L = codes.shape[-1]
    idx = np.zeros(codes.shape[:-1] + (L - k + 1,), dtype=np.int64)
    valid = np.ones_like(idx, dtype=bool)
    for j in range(k):
        window = codes[..., j: L - k + 1 + j]
        idx = idx * 4 + np.minimum(window, 3)
        valid &= window < N_CODE
    return np.where(valid, idx, -1)


def kmer_onehot(codes: np.ndarray, k: int = 3) -> np.ndarray:
    """Base codes -> binary k-mer matrix (the original "Grimm filter"), shape (..., L-k+1, 4**k)."""
    idx = kmer_indices(codes, k)
    out = np.zeros(idx.shape + (4 ** k,), dtype=np.uint8)
    np.put_along_axis(out, np.maximum(idx, 0)[..., None], 1, axis=-1)
    out[idx < 0] = 0
    return out


def kmer_profile(codes: np.ndarray, k: int = 3) -> np.ndarray:
    """Bag-of-k-mers count vector, used as a simple non-learned retrieval baseline."""
    return kmer_onehot(codes, k).sum(axis=-2).astype(np.float32)


def onehot_to_sequence(matrix: np.ndarray, k: int = 3) -> str:
    """Inverse of kmer_onehot for one read (replaces legacy reconstructSeqbyGF)."""
    matrix = np.asarray(matrix).reshape(matrix.shape[0], -1)
    rows_valid = matrix.any(axis=1)
    idx = matrix.argmax(axis=1)
    kmers = []
    for i, ok in zip(idx, rows_valid):
        if not ok:
            kmers.append(None)
            continue
        s, v = [], int(i)
        for _ in range(k):
            s.append(BASES[v % 4])
            v //= 4
        kmers.append("".join(reversed(s)))
    seq = [km[0] if km else "N" for km in kmers]
    last = kmers[-1] if kmers else None
    seq.extend(list(last[1:]) if last else ["N"] * (k - 1))
    return "".join(seq)
