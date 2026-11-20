import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

from ssgrl.data import AugmentConfig, augment, augment_numpy, kmer_onehot_tf, make_ssl_dataset
from ssgrl.encoding import (decode, encode, kmer_indices, kmer_onehot, onehot_to_sequence,
                            read_sequences, reverse_complement)
from ssgrl.simulate import simulate_reads

RNG = np.random.default_rng(0)


def random_reads(n=8, L=150):
    return RNG.integers(0, 4, (n, L)).astype(np.uint8)


def test_encode_decode_roundtrip():
    s = "ACGTNacgtX"
    assert decode(encode(s)) == "ACGTNACGTN"


def test_reverse_complement():
    assert decode(reverse_complement(encode("AACGTN"))) == "NACGTT"


def test_kmer_indices_and_onehot():
    idx = kmer_indices(encode("ACGTA"), 3)
    assert list(idx) == [0 * 16 + 1 * 4 + 2, 1 * 16 + 2 * 4 + 3, 2 * 16 + 3 * 4 + 0]
    assert list(kmer_indices(encode("ACNTA"), 3)) == [-1, -1, -1]
    m = kmer_onehot(encode("ACGTNACGT"), 3)
    assert m.shape == (7, 64)
    assert list(m.sum(1)) == [1, 1, 0, 0, 0, 1, 1]


def test_onehot_inverse():
    r = random_reads(3)
    for row in r:
        assert onehot_to_sequence(kmer_onehot(row)) == decode(row)


def test_tf_encoding_matches_numpy():
    r = random_reads(4)
    r[0, 10] = 4
    assert np.array_equal(kmer_onehot_tf(r.astype(np.int32)).numpy(), kmer_onehot(r))


def test_augment_tf_shapes_and_values():
    import tensorflow as tf
    c = tf.constant(random_reads(1)[0], tf.int32)
    for _ in range(20):
        out = augment(c, AugmentConfig()).numpy()
        assert out.shape == (150,) and out.min() >= 0 and out.max() <= 4


def test_augment_numpy_deterministic_and_noop():
    r = random_reads(5)
    a = augment_numpy(r, AugmentConfig(), seed=1)
    assert np.array_equal(a, augment_numpy(r, AugmentConfig(), seed=1))
    noop = AugmentConfig(sub_rate=0, rc_prob=0, crop_prob=0)
    assert np.array_equal(augment_numpy(r, noop), r)


def test_simulate_reads_from_reference():
    genome = random_reads(1, 5000)[0]
    reads, starts, strands = simulate_reads(genome, 50, error_rate=0.0, seed=0)
    for r, s, st in zip(reads, starts, strands):
        expected = genome[s:s + 150]
        assert np.array_equal(reverse_complement(r) if st else r, expected)


def test_read_sequences_fasta_and_fastq(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">a\nACGT\nAC\n>b\nGG\n")
    assert list(read_sequences(fa)) == [("a", "ACGTAC"), ("b", "GG")]
    fq = tmp_path / "x.fq"
    fq.write_text("@r1\nACGT\n+\nIIII\n@r2\nTT\n+\nII\n")
    assert list(read_sequences(fq)) == [("r1", "ACGT"), ("r2", "TT")]


@pytest.mark.parametrize("backbone", ["resnet1d", "resnet2d"])
def test_model_trains_and_saves(tmp_path, backbone):
    import keras
    from ssgrl.model import build_simsiam

    reads = random_reads(64)
    model = build_simsiam((148, 64), backbone=backbone, widths=(16, 32), blocks_per_stage=1,
                          proj_dim=32, pred_hidden=8)
    model.compile(optimizer=keras.optimizers.SGD(0.01, momentum=0.9))
    before = [w.numpy().copy() for w in model.backbone.trainable_weights]
    hist = model.fit(make_ssl_dataset(reads, 16), epochs=1, verbose=0)
    assert np.isfinite(hist.history["loss"][0])
    changed = [not np.allclose(a, b.numpy()) for a, b in zip(before, model.backbone.trainable_weights)]
    assert all(changed), "every backbone weight should receive a gradient"
    path = tmp_path / "bb.keras"
    model.backbone.save(path)
    loaded = keras.saving.load_model(path)
    x = kmer_onehot(reads[:4]).astype(np.float32)
    assert np.allclose(loaded.predict(x, verbose=0), model.backbone.predict(x, verbose=0), atol=1e-5)


def test_stop_gradient_in_loss():
    import tensorflow as tf
    from ssgrl.model import negative_cosine
    p = tf.Variable([[1.0, 0.0]])
    z = tf.Variable([[0.5, 0.5]])
    with tf.GradientTape() as tape:
        loss = negative_cosine(p, z)
    gp, gz = tape.gradient(loss, [p, z])
    assert gp is not None and gz is None
