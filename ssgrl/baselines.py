"""Reference baselines for read retrieval and binning.

Every baseline exposes the same interface as a trained encoder: given reads,
return one vector per read, compared by cosine similarity. Two exceptions are
distance-based and provide `rank(db, queries)` instead (MinHash, minimap2).

Implemented:
  * `kmer_profile`      - k-mer count vector; `l1` or `cosine` (Çelikkanat et al. 2024
                          report 4-mers with l1 as their best non-learned variant).
  * `tnf`               - tetranucleotide frequency exactly as DNABERT-S's
                          `calculate_tnf`: 4-mer counts over a window, divided by
                          the total. Equal to kmer_profile(k=4) after normalising.
  * `MinHashSketch`     - bottom-s sketch of canonical k-mers with the Mash
                          distance, D = -log(2j/(1+j))/k for Jaccard j.
  * `minimap2_rank`     - alignment with minimap2 through `mappy`, the
                          reference point a learned method has to justify itself against.
  * `DNABERT_S`         - transformers encoder, used only when the weights are
                          already on disk (this workspace cannot reach Hugging Face).

None of these need training, so all are evaluated on the same held-out reads as
the learned encoder.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .encoding import decode, kmer_indices, reverse_complement


def kmer_profile(reads: np.ndarray, k: int = 4, norm: str = "l1") -> np.ndarray:
    """(n, L) base codes -> (n, 4**k) k-mer count vector.

    norm: "l1" divides by the total count (frequencies), "none" keeps counts.
    Cosine similarity on an l1-normalised profile is the `KMER-l1` variant of
    Çelikkanat et al.; "none" with cosine is their `KMER-cosine`.
    """
    idx = kmer_indices(reads, k)
    out = np.zeros((len(reads), 4 ** k), dtype=np.float32)
    for i, row in enumerate(idx):
        valid = row[row >= 0]
        if len(valid):
            np.add.at(out[i], valid, 1.0)
    if norm == "l1":
        total = out.sum(axis=1, keepdims=True)
        out = np.divide(out, total, out=np.zeros_like(out), where=total > 0)
    elif norm != "none":
        raise ValueError(f"unknown norm {norm!r}")
    return out


def tnf(reads: np.ndarray) -> np.ndarray:
    """Tetranucleotide frequency, the `tnf` baseline of DNABERT-S."""
    return kmer_profile(reads, k=4, norm="l1")


def _rc_kmer_permutation(k: int) -> np.ndarray:
    """perm[i] = index of the reverse complement of k-mer i, in ACGT order."""
    perm = np.zeros(4 ** k, dtype=np.int64)
    for i in range(4 ** k):
        digits = [(i // 4 ** (k - 1 - j)) % 4 for j in range(k)]
        rc = [3 - d for d in reversed(digits)]
        perm[i] = sum(d * 4 ** (k - 1 - j) for j, d in enumerate(rc))
    return perm


def canonical_kmer_profile(reads: np.ndarray, k: int = 4, norm: str = "l1") -> np.ndarray:
    """Strand-folded k-mer profile: each k-mer is counted together with its reverse
    complement, so a read and its reverse complement give the same vector.

    This is the control for SSGRL's strand invariance. A plain profile scores
    about 0.03 on reverse-complemented queries while being the strongest method
    on substitutions, so without this baseline a learned encoder can appear to
    win purely by handling both strands - something canonical counting, standard
    in Mash and Kraken, already does for free.
    """
    counts = kmer_profile(reads, k, norm="none")
    perm = _rc_kmer_permutation(k)
    canon = np.minimum(np.arange(4 ** k), perm)
    folded = np.zeros((len(reads), 4 ** k), dtype=np.float32)
    np.add.at(folded.T, canon, counts.T)
    keep = np.unique(canon)
    folded = folded[:, keep]
    if norm == "l1":
        total = folded.sum(axis=1, keepdims=True)
        folded = np.divide(folded, total, out=np.zeros_like(folded), where=total > 0)
    elif norm != "none":
        raise ValueError(f"unknown norm {norm!r}")
    return folded


def canonical_kmer_hashes(read: np.ndarray, k: int = 21) -> np.ndarray:
    """64-bit hashes of the canonical (lexicographically smaller strand) k-mers."""
    fwd = kmer_indices(read[None, :], k)[0]
    rev = kmer_indices(reverse_complement(read)[None, :], k)[0]
    both = np.stack([fwd[fwd >= 0], rev[rev >= 0][::-1]]) if (fwd >= 0).sum() == (rev >= 0).sum() \
        else None
    if both is None:  # Ns break the 1:1 correspondence; fall back to the forward strand
        keep = fwd[fwd >= 0]
    else:
        keep = np.minimum(both[0], both[1])
    if not len(keep):
        return np.zeros(0, dtype=np.uint64)
    # SplitMix64-style finaliser: a cheap, well-mixed 64-bit hash
    x = keep.astype(np.uint64)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return np.unique(x)


@dataclass
class MinHashSketch:
    """Mash-style bottom-sketch comparison (Ondov et al. 2016).

    Defaults follow Mash's own for read-level comparison (k=21, s=1000); with
    150 bp reads a sketch holds every canonical k-mer, so the Jaccard index is exact.
    """

    k: int = 21
    sketch_size: int = 1000

    def sketch(self, reads: np.ndarray) -> list[np.ndarray]:
        return [canonical_kmer_hashes(r, self.k)[: self.sketch_size] for r in reads]

    def jaccard(self, a: np.ndarray, b: np.ndarray) -> float:
        if not len(a) or not len(b):
            return 0.0
        inter = len(np.intersect1d(a, b, assume_unique=True))
        union = len(a) + len(b) - inter
        return inter / union if union else 0.0

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Mash distance: -log(2j/(1+j))/k, 1.0 when nothing is shared."""
        j = self.jaccard(a, b)
        if j <= 0:
            return 1.0
        return float(-np.log(2 * j / (1 + j)) / self.k)

    def rank(self, db: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """(n_queries, k) indices into db, nearest first by Mash distance."""
        db_sk, q_sk = self.sketch(db), self.sketch(queries)
        out = np.zeros((len(queries), k), dtype=int)
        for i, qs in enumerate(q_sk):
            d = np.array([self.distance(qs, ds) for ds in db_sk])
            out[i] = np.argsort(d, kind="stable")[:k]
        return out


def minimap2_rank(db: np.ndarray, queries: np.ndarray, k: int = 10,
                  preset: str = "sr") -> np.ndarray:
    """Rank database reads by minimap2 alignment score (requires `mappy`).

    The database reads become the reference; each query is mapped against it.
    Ranking is by alignment score, then by number of matching bases. Queries with
    no hit get -1, which `recall_from_ranks` counts as a miss.
    """
    import tempfile

    import mappy

    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
        for i, r in enumerate(db):
            fh.write(f">{i}\n{decode(r).replace('N', '')}\n")
        ref_path = fh.name
    aligner = mappy.Aligner(ref_path, preset=preset, n_threads=2, best_n=max(k, 10))
    if not aligner:
        raise RuntimeError("failed to build the minimap2 index")
    out = np.full((len(queries), k), -1, dtype=int)
    for i, q in enumerate(queries):
        hits: dict[int, tuple] = {}
        for h in aligner.map(decode(q).replace("N", "")):
            key = int(h.ctg)
            score = (h.mlen, -h.NM if hasattr(h, "NM") else 0)
            if key not in hits or score > hits[key]:
                hits[key] = score
        if hits:
            order = sorted(hits, key=lambda c: hits[c], reverse=True)[:k]
            out[i, : len(order)] = order
    return out


def recall_from_ranks(ranks: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    """Recall@k when query i's correct answer is database item i."""
    truth = np.arange(len(ranks))[:, None]
    return {f"recall@{k}": float((ranks[:, :k] == truth).any(axis=1).mean()) for k in ks}


class DNABERT_S:
    """DNABERT-S embeddings (mean pooling over tokens), as in its own evaluation code.

    Loads from a local directory or model id. This workspace has no access to
    Hugging Face, so the benchmark skips this baseline unless weights are present.
    """

    def __init__(self, model_name_or_path: str = "zhihan1996/DNABERT-S",
                 model_max_length: int = 400, batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, model_max_length=model_max_length)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).eval()
        self.batch_size = batch_size

    def embed(self, reads: np.ndarray) -> np.ndarray:
        torch = self.torch
        out = []
        for start in range(0, len(reads), self.batch_size):
            seqs = [decode(r).replace("N", "") for r in reads[start:start + self.batch_size]]
            enc = self.tokenizer(seqs, return_tensors="pt", padding="longest", truncation=True)
            with torch.no_grad():
                hidden = self.model(enc["input_ids"], attention_mask=enc["attention_mask"])[0]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
        return np.concatenate(out)


def sequence_sha(reads: np.ndarray) -> str:
    """Short digest of a read set, so cached results can't be mixed up."""
    h = hashlib.sha256()
    for r in reads:
        h.update(r.tobytes())
    return h.hexdigest()[:12]
