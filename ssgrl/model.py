"""Self-supervised models (SimSiam, VICReg, Barlow Twins) for k-mer read matrices (Keras 3, TensorFlow backend).

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
    """Additive attention pooling over the position axis: (B, T, C) -> (B, C).

    Call with ``[x, mask]`` (mask: (B, T), 1 = real sequence) to ignore padded positions.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]
        self.proj.build(input_shape)
        self.score.build(tuple(input_shape[:-1]) + (self.units,))
        super().build(input_shape)

    def call(self, inputs):
        x, mask = (inputs if isinstance(inputs, (list, tuple)) else (inputs, None))
        scores = self.score(self.proj(x))  # (B, T, 1)
        if mask is not None:
            mask = keras.ops.expand_dims(keras.ops.cast(mask, scores.dtype), -1)
            scores = scores + (1.0 - mask) * -1e9
        weights = keras.ops.softmax(scores, axis=1)
        if mask is not None:  # an all-padding read would otherwise get uniform weights
            weights = weights * mask
        return keras.ops.sum(x * weights, axis=1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="ssgrl")
class PositionMask(layers.Layer):
    """(B, T, 4**k) k-mer matrix -> (B, ceil(T/stride) * repeat) mask of positions with sequence.

    A downsampled position is valid if any input row it covers holds a k-mer, so
    N padding from cropping (and reads shorter than read_len) is excluded from pooling.
    """

    def __init__(self, stride: int, repeat: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.stride, self.repeat = stride, repeat

    def call(self, x):
        valid = keras.ops.max(x, axis=-1, keepdims=True)  # (B, T, 1)
        if self.stride > 1:
            valid = keras.ops.max_pool(valid, self.stride, self.stride, padding="same")
        valid = keras.ops.squeeze(valid, -1)
        if self.repeat > 1:  # resnet2d: positions are flattened (H, W) cells, row-major
            valid = keras.ops.repeat(valid, self.repeat, axis=1)
        return valid

    def get_config(self):
        return {**super().get_config(), "stride": self.stride, "repeat": self.repeat}


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
                   rc_invariant: bool = True, masked_pooling: bool = True) -> keras.Model:
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
    total_stride = (2 if kind == "resnet2d" else 1) * 2 ** (len(widths) - 1)
    width_cells = x.shape[2] if kind == "resnet2d" else 1
    if kind == "resnet2d":
        x = layers.Reshape((-1, x.shape[-1]))(x)  # flatten positions x k-mer cells
    if masked_pooling:
        mask = PositionMask(total_stride, width_cells, name="position_mask")(inp)
        out = AttentionPooling(name="attention_pool")([x, mask])
    else:
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


def vicreg_loss(z1, z2, inv_w=25.0, var_w=25.0, cov_w=1.0, eps=1e-4):
    """VICReg (Bardes et al. 2022): invariance + variance hinge + covariance decorrelation."""
    inv = tf.reduce_mean(tf.square(z1 - z2))

    def var_cov(z):
        n = tf.cast(tf.shape(z)[0], tf.float32)
        d = tf.cast(tf.shape(z)[1], tf.float32)
        z = z - tf.reduce_mean(z, axis=0)
        std = tf.sqrt(tf.reduce_mean(tf.square(z), axis=0) + eps)
        var = tf.reduce_mean(tf.nn.relu(1.0 - std))
        cov = tf.matmul(z, z, transpose_a=True) / (n - 1.0)
        off = tf.reduce_sum(tf.square(cov)) - tf.reduce_sum(tf.square(tf.linalg.diag_part(cov)))
        return var, off / d

    v1, c1 = var_cov(z1)
    v2, c2 = var_cov(z2)
    return inv_w * inv + var_w * 0.5 * (v1 + v2) + cov_w * (c1 + c2)


def barlow_twins_loss(z1, z2, lambd=None, eps=1e-5):
    """Barlow Twins (Zbontar et al. 2021): cross-correlation matrix pushed towards identity."""
    n = tf.cast(tf.shape(z1)[0], tf.float32)
    d = tf.cast(tf.shape(z1)[1], tf.float32)
    lambd = 1.0 / d if lambd is None else lambd

    def standardize(z):
        z = z - tf.reduce_mean(z, axis=0)
        return z / (tf.math.reduce_std(z, axis=0) + eps)

    c = tf.matmul(standardize(z1), standardize(z2), transpose_a=True) / n
    on = tf.reduce_sum(tf.square(tf.linalg.diag_part(c) - 1.0))
    off = tf.reduce_sum(tf.square(c)) - tf.reduce_sum(tf.square(tf.linalg.diag_part(c)))
    return (on + lambd * off) / d  # /d keeps the scale comparable across head sizes


LOSSES = ("simsiam", "vicreg", "barlow")


class SSLModel(keras.Model):
    """Two-view self-supervised model. loss: simsiam (needs predictor), vicreg or barlow."""

    def __init__(self, backbone, projector, predictor=None, loss: str = "simsiam", **kwargs):
        super().__init__(**kwargs)
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if loss == "simsiam" and predictor is None:
            raise ValueError("simsiam needs a predictor")
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor
        self.loss_name = loss
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
        if self.loss_name == "simsiam":
            p1 = self.predictor(z1, training=training)
            p2 = self.predictor(z2, training=training)
            loss = 0.5 * negative_cosine(p1, z2) + 0.5 * negative_cosine(p2, z1)
        elif self.loss_name == "vicreg":
            loss = vicreg_loss(z1, z2)
        else:
            loss = barlow_twins_loss(z1, z2)
        return loss, z1

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss, z1 = self._forward(data, training=True)
            reg = tf.add_n(self.losses) if self.losses else 0.0
            total = loss + reg
        variables = self.trainable_variables
        grads = tape.gradient(total, variables)
        self.optimizer.apply_gradients(
            (g, v) for g, v in zip(grads, variables) if g is not None)
        self.loss_tracker.update_state(loss)
        self.std_tracker.update_state(collapse_score(z1))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        loss, z1 = self._forward(data, training=False)
        self.loss_tracker.update_state(loss)
        self.std_tracker.update_state(collapse_score(z1))
        return {m.name: m.result() for m in self.metrics}


SimSiam = SSLModel  # backwards-compatible name


def build_expander(in_dim: int, dim: int = 1024) -> keras.Model:
    """VICReg/Barlow projector: Dense-BN-ReLU x2 then a linear layer."""
    inp = keras.Input((in_dim,))
    x = inp
    for _ in range(2):
        x = layers.Dense(dim, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    out = layers.Dense(dim, use_bias=False)(x)
    return keras.Model(inp, out, name="expander")


def build_ssl_model(input_shape, loss: str = "simsiam", backbone: str = "resnet1d",
                    widths=(64, 128, 256), blocks_per_stage: int = 2, proj_dim: int = 256,
                    pred_hidden: int = 64, weight_decay: float = 5e-4,
                    rc_invariant: bool = True, masked_pooling: bool = True) -> SSLModel:
    """proj_dim is the SimSiam projector size, or the expander size for vicreg/barlow."""
    bb = build_backbone(input_shape, backbone, widths, blocks_per_stage, weight_decay, rc_invariant,
                        masked_pooling)
    if loss == "simsiam":
        proj = build_projector(bb.output_shape[-1], proj_dim)
        pred = build_predictor(proj_dim, pred_hidden)
    else:
        proj, pred = build_expander(bb.output_shape[-1], proj_dim), None
    model = SSLModel(bb, proj, pred, loss=loss, name=f"ssl_{loss}")
    # build every sub-model so weights can be loaded straight away
    model(tf.zeros((1,) + tuple(input_shape)))
    if pred is not None:
        pred(tf.zeros((1, proj_dim)))
    return model


def build_simsiam(input_shape, backbone: str = "resnet1d", widths=(64, 128, 256),
                  blocks_per_stage: int = 2, proj_dim: int = 256, pred_hidden: int = 64,
                  weight_decay: float = 5e-4, rc_invariant: bool = True) -> SSLModel:
    return build_ssl_model(input_shape, "simsiam", backbone, widths, blocks_per_stage,
                           proj_dim, pred_hidden, weight_decay, rc_invariant)
