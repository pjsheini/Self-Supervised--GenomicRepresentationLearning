"""tf.data pipelines and sequence-level augmentations.

Augmentations act on the bases *before* k-mer encoding, so they mimic real
variation between reads of the same locus:
  * random substitutions (sequencing error / SNPs)
  * reverse complement (the other strand of the same fragment)
  * random crop + shift (partially overlapping reads), remainder filled with N

The legacy code instead rolled the matrix along the k-mer axis, which
relabels every k-mer and destroys the sequence identity it should preserve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

N_CODE = 4


@dataclass
class AugmentConfig:
    sub_rate: float = 0.02      # per-base substitution probability
    rc_prob: float = 0.5        # probability of reverse-complementing
    min_keep: float = 0.7       # crop keeps at least this fraction of the read
    crop_prob: float = 0.8


def _reverse_complement(codes: tf.Tensor) -> tf.Tensor:
    comp = tf.where(codes < N_CODE, 3 - codes, codes)
    return tf.reverse(comp, axis=[-1])


def augment(codes: tf.Tensor, cfg: AugmentConfig) -> tf.Tensor:
    """codes: (L,) int32 base codes -> augmented (L,) int32."""
    L = tf.shape(codes)[0]
    # substitutions (only on real bases)
    if cfg.sub_rate > 0:
        hit = tf.random.uniform([L]) < cfg.sub_rate
        offset = tf.random.uniform([L], 1, 4, dtype=tf.int32)  # guarantees a *different* base
        mutated = (codes + offset) % 4
        codes = tf.where(hit & (codes < N_CODE), mutated, codes)
    # reverse complement
    if cfg.rc_prob > 0:
        codes = tf.cond(tf.random.uniform([]) < cfg.rc_prob,
                        lambda: _reverse_complement(codes), lambda: codes)
    # crop a window and place it at a random offset, pad with N
    if cfg.crop_prob > 0 and cfg.min_keep < 1.0:
        def crop(c):
            min_len = tf.cast(tf.math.ceil(cfg.min_keep * tf.cast(L, tf.float32)), tf.int32)
            keep = tf.random.uniform([], min_len, L + 1, dtype=tf.int32)
            start = tf.random.uniform([], 0, L - keep + 1, dtype=tf.int32)
            window = c[start:start + keep]
            pos = tf.random.uniform([], 0, L - keep + 1, dtype=tf.int32)
            return tf.concat([tf.fill([pos], N_CODE), window,
                              tf.fill([L - keep - pos], N_CODE)], axis=0)
        codes = tf.cond(tf.random.uniform([]) < cfg.crop_prob, lambda: crop(codes), lambda: codes)
    return codes


def kmer_onehot_tf(codes: tf.Tensor, k: int = 3) -> tf.Tensor:
    """(..., L) int32 codes -> (..., L-k+1, 4**k) float32 one-hot k-mer matrix (zeros where N)."""
    L = tf.shape(codes)[-1]
    n = L - k + 1
    idx = tf.zeros_like(codes[..., :n])
    valid = tf.ones_like(idx, dtype=tf.bool)
    for j in range(k):
        w = codes[..., j:j + n]
        idx = idx * 4 + tf.minimum(w, 3)
        valid &= w < N_CODE
    idx = tf.where(valid, idx, -1)  # one_hot of -1 is all zeros
    return tf.one_hot(idx, 4 ** k, dtype=tf.float32)


def make_ssl_dataset(reads: np.ndarray, batch_size: int, k: int = 3,
                     aug: AugmentConfig | None = None, shuffle: bool = True,
                     seed: int | None = None) -> tf.data.Dataset:
    """Dataset of (view1, view2) pairs, each (B, L-k+1, 4**k)."""
    aug = aug or AugmentConfig()
    ds = tf.data.Dataset.from_tensor_slices(reads.astype(np.int32))
    if shuffle:
        ds = ds.shuffle(min(len(reads), 50_000), seed=seed, reshuffle_each_iteration=True)

    def two_views(c):
        return (kmer_onehot_tf(augment(c, aug), k), kmer_onehot_tf(augment(c, aug), k))

    return (ds.map(two_views, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))


def make_embed_dataset(reads: np.ndarray, batch_size: int = 512, k: int = 3) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(reads.astype(np.int32))
    return (ds.map(lambda c: kmer_onehot_tf(c, k), num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def augment_numpy(reads: np.ndarray, cfg: AugmentConfig, seed: int = 0) -> np.ndarray:
    """Deterministic NumPy twin of `augment`, used to build reproducible eval queries."""
    rng = np.random.default_rng(seed)
    out = reads.astype(np.int64).copy()
    n, L = out.shape
    for i in range(n):
        c = out[i]
        hit = (rng.random(L) < cfg.sub_rate) & (c < N_CODE)
        c[hit] = (c[hit] + rng.integers(1, 4, hit.sum())) % 4
        if rng.random() < cfg.rc_prob:
            c = np.where(c < N_CODE, 3 - c, c)[::-1]
        if rng.random() < cfg.crop_prob and cfg.min_keep < 1.0:
            keep = rng.integers(int(np.ceil(cfg.min_keep * L)), L + 1)
            start = rng.integers(0, L - keep + 1)
            pos = rng.integers(0, L - keep + 1)
            new = np.full(L, N_CODE)
            new[pos:pos + keep] = c[start:start + keep]
            c = new
        out[i] = c
    return out.astype(np.uint8)
