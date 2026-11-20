"""SimSiam model for k-mer read matrices (Keras 3, TensorFlow backend).

Backbones
---------
* ``resnet2d`` - the original idea: treat the (positions x k-mers) matrix as a
  1-channel image and run a small pre-activation ResNet over it. This replaces
  the `resnet_cifar10_v2` module the legacy code imported but never shipped.
* ``resnet1d`` - treats the 4**k k-mer columns as channels and convolves along
  the read only. Usually the better fit, since neighbouring k-mer columns have
  no spatial meaning.

Both end in attention pooling over read positions (the legacy code applied its
attention layer to a reshaped feature vector instead, which doesn't attend
over anything meaningful).

Fixes versus legacy SSLModel.py
-------------------------------
* stop-gradient on the target branch (SimSiam collapses without it)
* symmetric loss 0.5*D(p1,z2) + 0.5*D(p2,z1); dropped the stray D(p1,z1) term
* gradients zipped with the variables they were computed for
* no Rescaling(1/127.5, -1) (an image leftover; inputs are 0/1)
* projector ends in BN, not ReLU; predictor is the SimSiam bottleneck MLP
* the meaningless Accuracy(p1, p2) metric replaced by a collapse monitor
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable(package="ssgrl")
class AttentionPooling(layers.Layer):
    """Additive attention pooling over the position axis: (B, T, C) -> (B, C)."""

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.score.build(tuple(input_shape[:-1]) + (self.units,))
        super().build(input_shape)

    def call(self, x):
        weights = keras.ops.softmax(self.score(self.proj(x)), axis=1)  # (B, T, 1)
        return keras.ops.sum(x * weights, axis=1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def _rc_kmer_permutation(k: int) -> list[int]:
    """perm[i] = column index of the reverse complement of k-mer i (ACGT order)."""
    perm = []
    for i in range(4 ** k):
        digits = [(i // 4 ** (k - 1 - j)) % 4 for j in range(k)]
        rc = [3 - d for d in reversed(digits)]
        perm.append(sum(d * 4 ** (k - 1 - j) for j, d in enumerate(rc)))
    return perm


@keras.saving.register_keras_serializable(package="ssgrl")
class ReverseComplementKmers(layers.Layer):
    """Maps a k-mer matrix of a read to the k-mer matrix of its reverse complement."""

    def __init__(self, k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.perm = _rc_kmer_permutation(k)

    def call(self, x):
        return keras.ops.take(keras.ops.flip(x, axis=1), self.perm, axis=-1)

    def get_config(self):
        return {**super().get_config(), "k": self.k}


def _preact_block(x, filters, stride, conv, wd):
    reg = keras.regularizers.l2(wd) if wd else None
    shortcut = x
    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    if stride != 1 or x.shape[-1] != filters:
        shortcut = conv(filters, 1, strides=stride, use_bias=False, kernel_regularizer=reg)(y)
    y = conv(filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = conv(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg)(y)
    return layers.Add()([shortcut, y])


def build_backbone(input_shape, kind: str = "resnet1d", widths=(64, 128, 256),
                   blocks_per_stage: int = 2, weight_decay: float = 0.0,
                   rc_invariant: bool = True) -> keras.Model:
    """(T, 4**k) k-mer matrix -> pooled representation of size widths[-1]."""
    inp = keras.Input(shape=input_shape, name="kmer_matrix")
    if kind == "resnet2d":
        conv = layers.Conv2D
        x = layers.Reshape(input_shape + (1,))(inp)
        # strided stem: the full-resolution 148x64 grid is very expensive on CPU
        x = conv(widths[0] // 2, 3, strides=2, padding="same", use_bias=False)(x)
    elif kind == "resnet1d":
        conv = layers.Conv1D
        x = conv(widths[0], 5, padding="same", use_bias=False)(inp)
    else:
        raise ValueError(f"unknown backbone {kind!r}")
    for stage, w in enumerate(widths):
        for b in range(blocks_per_stage):
            stride = 2 if (stage > 0 and b == 0) else 1
            x = _preact_block(x, w, stride, conv, weight_decay)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if kind == "resnet2d":
        x = layers.Reshape((-1, x.shape[-1]))(x)  # flatten positions x k-mer cells
    out = AttentionPooling(name="attention_pool")(x)
    core = keras.Model(inp, out, name=f"backbone_{kind}")
    if not rc_invariant:
        return core
    # strand invariance by construction: average the embeddings of both strands
    inp2 = keras.Input(shape=input_shape, name="kmer_matrix")
    k = {16: 2, 64: 3, 256: 4, 1024: 5}[input_shape[-1]]
    fwd = core(inp2)
    rev = core(ReverseComplementKmers(k)(inp2))
    out2 = layers.Average()([fwd, rev])
    return keras.Model(inp2, out2, name=f"backbone_{kind}_rc")


def build_projector(in_dim: int, proj_dim: int = 256) -> keras.Model:
    inp = keras.Input((in_dim,))
    x = layers.Dense(proj_dim, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(proj_dim, use_bias=False)(x)
    out = layers.BatchNormalization()(x)
    return keras.Model(inp, out, name="projector")


def build_predictor(proj_dim: int = 256, hidden: int = 64) -> keras.Model:
    inp = keras.Input((proj_dim,))
    x = layers.Dense(hidden, use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    out = layers.Dense(proj_dim)(x)
    return keras.Model(inp, out, name="predictor")


def negative_cosine(p, z):
    """SimSiam D(p, z) with stop-gradient on z."""
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(p * z, axis=1))


def collapse_score(z):
    """std of l2-normalised embeddings across the batch, scaled by sqrt(dim).

    ~1.0 means healthy spread; values near 0 mean every input maps to the same point.
    """
    z = tf.math.l2_normalize(z, axis=1)
    d = tf.cast(tf.shape(z)[1], tf.float32)
    return tf.reduce_mean(tf.math.reduce_std(z, axis=0)) * tf.sqrt(d)


class SimSiam(keras.Model):
    def __init__(self, backbone, projector, predictor, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.std_tracker = keras.metrics.Mean(name="z_std")

    @property
    def metrics(self):
        return [self.loss_tracker, self.std_tracker]

    def call(self, x, training=False):
        return self.projector(self.backbone(x, training=training), training=training)

    def _forward(self, data, training):
        x1, x2 = data
        z1, z2 = self(x1, training=training), self(x2, training=training)
        p1 = self.predictor(z1, training=training)
        p2 = self.predictor(z2, training=training)
        loss = 0.5 * negative_cosine(p1, z2) + 0.5 * negative_cosine(p2, z1)
        return loss, z1

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, z1 = self._forward(data, training=True)
            reg = tf.add_n(self.losses) if self.losses else 0.0
            total = loss + reg
        variables = self.trainable_variables
        grads = tape.gradient(total, variables)
        self.optimizer.apply_gradients(zip(grads, variables))
        self.loss_tracker.update_state(loss)
        self.std_tracker.update_state(collapse_score(z1))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        loss, z1 = self._forward(data, training=False)
        self.loss_tracker.update_state(loss)
        self.std_tracker.update_state(collapse_score(z1))
        return {m.name: m.result() for m in self.metrics}


def build_simsiam(input_shape, backbone: str = "resnet1d", widths=(64, 128, 256),
                  blocks_per_stage: int = 2, proj_dim: int = 256, pred_hidden: int = 64,
                  weight_decay: float = 5e-4, rc_invariant: bool = True) -> SimSiam:
    bb = build_backbone(input_shape, backbone, widths, blocks_per_stage, weight_decay, rc_invariant)
    proj = build_projector(bb.output_shape[-1], proj_dim)
    pred = build_predictor(proj_dim, pred_hidden)
    return SimSiam(bb, proj, pred, name="simsiam")
