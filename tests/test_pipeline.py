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


@pytest.mark.parametrize("backbone,loss", [("resnet1d", "simsiam"), ("resnet2d", "simsiam"),
                                          ("resnet1d", "vicreg"), ("resnet1d", "barlow")])
def test_model_trains_and_saves(tmp_path, backbone, loss):
    import keras
    from ssgrl.model import build_ssl_model

    reads = random_reads(64)
    model = build_ssl_model((148, 64), loss=loss, backbone=backbone, widths=(16, 32),
                            blocks_per_stage=1, proj_dim=32, pred_hidden=8)
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


def test_reverse_complement_layer_matches_encoding():
    from ssgrl.model import ReverseComplementKmers
    r = random_reads(4)
    r[1, 20] = 4
    x = kmer_onehot(r).astype(np.float32)
    x_rc = kmer_onehot(np.stack([reverse_complement(row) for row in r])).astype(np.float32)
    assert np.array_equal(ReverseComplementKmers(3)(x).numpy(), x_rc)


@pytest.mark.parametrize("backbone", ["resnet1d", "resnet2d"])
def test_backbone_is_strand_invariant(backbone):
    from ssgrl.model import build_backbone
    bb = build_backbone((148, 64), backbone, widths=(16, 32), blocks_per_stage=1)
    r = random_reads(4)
    x = kmer_onehot(r).astype(np.float32)
    x_rc = kmer_onehot(np.stack([reverse_complement(row) for row in r])).astype(np.float32)
    assert np.allclose(bb.predict(x, verbose=0), bb.predict(x_rc, verbose=0), atol=1e-5)


def test_vicreg_and_barlow_penalise_collapse():
    import tensorflow as tf
    from ssgrl.model import barlow_twins_loss, vicreg_loss
    rng = np.random.default_rng(0)
    spread = tf.constant(rng.normal(size=(256, 32)), tf.float32)
    collapsed = tf.ones((256, 32)) + 1e-3 * tf.constant(rng.normal(size=(256, 32)), tf.float32)
    # identical views: invariance term is zero, so only collapse is penalised
    assert float(vicreg_loss(spread, spread)) < float(vicreg_loss(collapsed, collapsed))
    assert float(vicreg_loss(collapsed, collapsed)) > 20  # variance hinge ~ 25 * 1
    assert float(barlow_twins_loss(spread, spread)) < 0.2
    assert float(barlow_twins_loss(spread, tf.constant(rng.normal(size=(256, 32)), tf.float32))) > 0.8


def test_effective_rank():
    from ssgrl.evaluate import effective_rank
    rng = np.random.default_rng(0)
    assert effective_rank(rng.normal(size=(2000, 16))) > 14
    low = rng.normal(size=(2000, 1)) @ rng.normal(size=(1, 16))
    assert effective_rank(low) < 1.5


@pytest.mark.parametrize("backbone", ["resnet1d", "resnet2d"])
def test_masked_pooling_ignores_padding(backbone):
    """An all-N read must not leak into pooling, and masks must line up with features."""
    from ssgrl.model import build_backbone
    bb = build_backbone((148, 64), backbone, widths=(16, 32, 64), blocks_per_stage=1)
    mask_layer = [l for l in bb.layers if l.name.startswith("backbone_")][0].get_layer("position_mask")
    r = random_reads(2)
    r[0, 100:] = 4
    m = mask_layer(kmer_onehot(r).astype(np.float32)).numpy()
    assert m[1].all() and not m[0].all() and m[0].any()
    empty = kmer_onehot(np.full((1, 150), 4, np.uint8)).astype(np.float32)
    assert np.allclose(bb.predict(empty, verbose=0), 0.0)


def test_unique_reads_collapses_strands():
    from ssgrl.evaluate import unique_reads
    r = random_reads(3)
    db = np.stack([r[0], r[1], reverse_complement(r[0]), r[2], r[1]])
    assert len(unique_reads(db)) == 3


# --- baselines -------------------------------------------------------------

def test_kmer_profile_and_tnf():
    from ssgrl.baselines import kmer_profile, tnf
    r = random_reads(4)
    p = kmer_profile(r, 4, "l1")
    assert p.shape == (4, 256)
    assert np.allclose(p.sum(1), 1.0)
    assert np.allclose(p, tnf(r))
    counts = kmer_profile(r, 4, "none")
    assert counts.sum(1).min() == 150 - 3


def test_canonical_profile_is_strand_invariant():
    from ssgrl.baselines import canonical_kmer_profile
    r = random_reads(5)
    rc = np.stack([reverse_complement(row) for row in r])
    a = canonical_kmer_profile(r, 4)
    assert a.shape == (5, 136)  # 256 4-mers fold to 136 canonical pairs
    assert np.allclose(a, canonical_kmer_profile(rc, 4))
    # a plain profile is NOT strand invariant; that is the point of this control
    from ssgrl.baselines import kmer_profile
    assert not np.allclose(kmer_profile(r, 4), kmer_profile(rc, 4))


def test_minhash_strand_invariant_and_self_distance():
    from ssgrl.baselines import MinHashSketch, canonical_kmer_hashes
    r = random_reads(3)
    for row in r:
        assert np.array_equal(canonical_kmer_hashes(row), canonical_kmer_hashes(reverse_complement(row)))
    mh = MinHashSketch()
    a = canonical_kmer_hashes(r[0])
    assert mh.distance(a, a) == 0.0
    assert mh.jaccard(a, canonical_kmer_hashes(r[1])) < 0.5


def test_minhash_and_minimap2_retrieve_exact_copies():
    from ssgrl.baselines import MinHashSketch, minimap2_rank, recall_from_ranks
    db = random_reads(20)
    assert recall_from_ranks(MinHashSketch().rank(db, db, 1), (1,))["recall@1"] == 1.0
    assert recall_from_ranks(minimap2_rank(db, db, 1), (1,))["recall@1"] == 1.0


def test_recall_from_ranks_handles_unmapped():
    from ssgrl.baselines import recall_from_ranks
    ranks = np.array([[0, 1], [-1, -1], [5, 2]])
    assert recall_from_ranks(ranks, (1,))["recall@1"] == 1 / 3


# --- binning ---------------------------------------------------------------

def test_kmedoid_recovers_well_separated_clusters():
    from ssgrl.binning import bin_quality, kmedoid, normalize
    rng = np.random.default_rng(0)
    centers = np.eye(3) * 10
    emb = np.repeat(centers, 40, axis=0) + rng.normal(0, 0.1, (120, 3))
    labels = kmedoid(normalize(emb), min_similarity=0.9, min_bin_size=5)
    assert len(np.unique(labels[labels != -1])) == 3
    q = bin_quality(np.repeat(np.arange(3), 40), labels)
    assert q["f1_bins"]["0.9"] == 3 and q["ari"] > 0.99


def test_threshold_percentile_is_between_min_and_max():
    from ssgrl.binning import class_center_similarity_threshold, normalize
    rng = np.random.default_rng(0)
    emb = normalize(rng.normal(size=(200, 8)))
    labels = rng.integers(0, 4, 200)
    t = class_center_similarity_threshold(emb, labels)
    assert -1.0 <= t <= 1.0


def test_tiling_and_pooling_round_trip():
    from ssgrl.baselines import canonical_kmer_profile
    from ssgrl.binning import pooled_embedding, tile_sequences
    seqs = ["".join("ACGT"[i % 4] for i in range(450)), "".join("ACGT"[i % 3] for i in range(300))]
    tiles, owner = tile_sequences(seqs, 150)
    assert tiles.shape == (5, 150) and list(owner) == [0, 0, 0, 1, 1]
    pooled = pooled_embedding(lambda r: canonical_kmer_profile(r, 4), tiles, owner, 2)
    assert pooled.shape == (2, 136)
    # pooling the tiles of one sequence equals the mean of its tile profiles
    expected = canonical_kmer_profile(tiles[:3], 4).mean(axis=0)
    assert np.allclose(pooled[0], expected, atol=1e-6)


def test_simulated_metagenome_round_trip(tmp_path):
    from ssgrl.binning import read_tsv
    from ssgrl.simulate_metagenome import main as sim_main
    g1, g2 = tmp_path / "a.fa", tmp_path / "b.fa"
    rng = np.random.default_rng(0)
    for path in (g1, g2):
        path.write_text(">x\n" + decode(rng.integers(0, 4, 20000).astype(np.uint8)) + "\n")
    out = tmp_path / "meta"
    sim_main([str(g1), str(g2), "-o", str(out), "-n", "100", "-l", "150"])
    seqs, labels = read_tsv(out / "binning_5.tsv")
    assert len(seqs) == len(labels) >= 40
    assert set(labels) == {"a", "b"}
    assert all(len(s) == 150 for s in seqs)
