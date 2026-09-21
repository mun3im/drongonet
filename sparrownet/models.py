"""
models.py — SparrowNet and the drongonet-micro baseline.

Two architectures:
  build_drongonet_micro()  faithful port of drongonet's 6b_micro_final.py
                           (FreqEmphasis -> Conv6 -> MaxPool -> Conv12 -> Conv12_1x1
                            -> GAP -> Dropout -> Dense2). 919 params. The baseline that
                           answers "is the ASEAN result a dataset or an architecture
                           problem" before we change anything.
  build_sparrownet()       local receptive field -> per-frame logits -> global MAX,
                           after the sparrow model in Grill & Schluter 2017 (Table II).

Note on the port: drongonet's own docstring says "SeparableConv2D" but the code uses
plain Conv2D. The code is what shipped (919 params confirms it), so the port follows
the code.
"""

import numpy as np
import tensorflow as tf

L = tf.keras.layers


class FrequencyEmphasis(L.Layer):
    """Learnable per-mel-bin gate: sigmoid(freq_weights * scale), multiplied into the
    spectrogram. freq_bins + 1 params. Ported verbatim from drongonet."""

    def __init__(self, freq_bins=16, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins
        self.freq_weights = self.add_weight(
            name="frequency_weights", shape=(1, 1, freq_bins, 1),
            initializer=tf.constant_initializer(1.0), trainable=True, dtype=tf.float32)
        self.scale = self.add_weight(
            name="scale", shape=(1,),
            initializer=tf.constant_initializer(3.0), trainable=True, dtype=tf.float32)

    def call(self, inputs, training=None):
        return inputs * tf.math.sigmoid(self.freq_weights * self.scale)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"freq_bins": self.freq_bins})
        return cfg


def build_drongonet_micro(input_shape=(184, 16, 1), num_classes=2, dropout=0.1,
                          batch_norm=False):
    """drongonet-micro. 919 params at (184,16,1) in its faithful form.

    batch_norm defaults to False because that is what drongonet actually ships, and
    this is the baseline. Set it True for the *control* arm: batch norm alone was worth
    ~0.13 cross-corpus AUC on SparrowNet, so comparing a BN'd SparrowNet against a
    BN-less micro would credit the topology for a gain that normalization supplied.
    The control separates the two.
    """
    inputs = L.Input(shape=input_shape)
    x = FrequencyEmphasis(freq_bins=input_shape[1], name="frequency_emphasis")(inputs)

    def block(x, filters, kernel, name):
        x = L.Conv2D(filters, kernel, padding="same",
                     activation=None if batch_norm else "relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=name)(x)
        if batch_norm:
            x = L.BatchNormalization(name=f"{name}_bn")(x)
            x = L.ReLU(name=f"{name}_relu")(x)
        return x

    x = block(x, 6, (3, 3), "conv1")
    x = L.MaxPooling2D((2, 2))(x)
    x = block(x, 12, (3, 3), "conv2")
    x = block(x, 12, (1, 1), "pointwise_conv")
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs,
                          name="drongonet_micro_bn" if batch_norm else "drongonet_micro")


def time_receptive_field(n_time_stages: int, kernel_t: int = 3, head_kernel_t: int = 1) -> int:
    """Receptive field in frames for the sparrownet stack below.
    RF_l = RF_{l-1} + (k-1)*jump_{l-1}, jump_l = jump_{l-1} * stride_l.
    Stage 1 is stride-2 conv; stages 2..n are depthwise stride-2; then the head's
    time-kernel is applied at the final jump."""
    rf, jump = 1, 1
    for _ in range(n_time_stages):
        rf += (kernel_t - 1) * jump
        jump *= 2
    rf += (head_kernel_t - 1) * jump
    return rf


def build_sparrownet(input_shape=(622, 16, 1), num_classes=2,
                     n_time_stages=4, head_kernel_t=1, width=(8, 16),
                     dropout=0.1, pool="max", batch_norm=True, name="sparrownet"):
    """
    Local-window detector: a short-receptive-field fully-convolutional trunk emits a
    logit per time position, then ONE global max over time decides the clip.

    Why max and not average: the DCASE label is "at least one bird event somewhere in
    the clip" (the MI assumption). Max matches that directly, and the paper's Fig. 4
    shows global-mean training "strongly impairs discrimination". `pool` is exposed so
    that claim is ablatable rather than assumed.

    n_time_stages controls the receptive field (see time_receptive_field):
        3 -> 15 frames    4 -> 31      5 -> 63      6 -> 127 frames (~2.0s)
    Frequency is strided down alongside time for the first log2(n_mels) stages, then
    collapsed by a full-height depthwise conv so the trunk is time-only afterwards.

    batch_norm is on by default and is load-bearing, not cosmetic: without it every
    n_time_stages>=6 model collapsed to a constant output (AUC exactly 0.5000) because
    the signal vanishes through six undamped depthwise stages. The paper does the same
    thing for this architecture ("In sparrow, we also apply batch normalization to all
    layers"), and TFLite folds BN into the preceding conv at conversion, so it costs
    essentially nothing at inference.
    """
    c1, c2 = width
    n_mels = input_shape[1]
    inputs = L.Input(shape=input_shape)
    x = FrequencyEmphasis(freq_bins=n_mels, name="frequency_emphasis")(inputs)

    def maybe_bn(x, name):
        return L.BatchNormalization(name=name)(x) if batch_norm else x

    # Stage 1: full-width strided conv. Early layers stay wide (drongonet lesson:
    # narrowing the early layer is the one change that reliably breaks recall).
    x = L.Conv2D(c1, (3, 3), strides=(2, 2), padding="same",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="stem")(x)
    x = maybe_bn(x, "stem_bn")
    x = L.ReLU(name="stem_relu")(x)
    freq = int(np.ceil(n_mels / 2))

    # Stages 2..n: depthwise-separable, stride 2 in BOTH dims throughout.
    # Strides are deliberately kept equal: unequal depthwise strides like (2,1) are
    # rejected outright ("only supports equal length strides in the row and column
    # dimensions") on CPU and are a portability risk on TFLite Micro, which is the
    # deployment target. Once frequency has collapsed to 1, stride 2 with 'same'
    # padding is a no-op on that axis (ceil(1/2) == 1), so equal strides cost nothing.
    for i in range(2, n_time_stages + 1):
        x = L.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", name=f"dw{i}")(x)
        x = maybe_bn(x, f"dw{i}_bn")
        x = L.ReLU(name=f"dw{i}_relu")(x)
        x = L.Conv2D(c2, (1, 1), padding="same", name=f"pw{i}")(x)
        x = maybe_bn(x, f"pw{i}_bn")
        x = L.ReLU(name=f"pw{i}_relu")(x)
        freq = int(np.ceil(freq / 2))

    # Collapse any remaining frequency extent so the head is purely temporal.
    if freq > 1:
        x = L.DepthwiseConv2D((1, freq), padding="valid", name="freq_collapse")(x)
        freq = 1

    # Local classifier head: one logit per surviving time position.
    if head_kernel_t > 1:
        x = L.DepthwiseConv2D((head_kernel_t, 1), padding="same", name="head_dw")(x)
    x = L.Dropout(dropout)(x)
    x = L.Conv2D(num_classes, (1, 1), padding="same", name="local_logits")(x)

    # Pool the local decisions over time, then softmax once.
    #
    # Pools in 2D over (time, freq) rather than Reshape((-1, C)) + 1D pooling. The
    # reshape carried a -1, which made the converted graph emit SHAPE / STRIDED_SLICE /
    # PACK and four dynamic-shape tensors — TFLite Micro requires static shapes, so
    # that form is not deployable on the Cortex-M4 target. Frequency is already
    # collapsed to 1 here, so pooling over (T,1) is numerically the same operation and
    # converts to a single REDUCE_MAX with static shapes.
    if pool == "max":
        x = L.GlobalMaxPooling2D(name="global_max")(x)
    elif pool == "avg":
        x = L.GlobalAveragePooling2D(name="global_avg")(x)
    else:
        raise ValueError(f"pool must be 'max' or 'avg', got {pool!r}")
    outputs = L.Softmax(name="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    model.receptive_field_frames = time_receptive_field(n_time_stages, 3, head_kernel_t)
    return model


def focal_loss(gamma=2.0, alpha=0.5):
    """Categorical focal loss, drongonet's settings (gamma=2, alpha=0.5)."""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss


if __name__ == "__main__":
    print("receptive field by n_time_stages (frames @ 16ms/frame):")
    for n in range(3, 8):
        rf = time_receptive_field(n)
        print(f"  n_time_stages={n}: {rf:4d} frames = {rf*256/16000:.2f}s")

    print()
    m = build_drongonet_micro()
    print(f"{m.name:22s} params={m.count_params():6d} in={m.input_shape} out={m.output_shape}")

    for n in (4, 5, 6):
        m = build_sparrownet(n_time_stages=n)
        print(f"sparrownet(stages={n})    params={m.count_params():6d} "
              f"RF={m.receptive_field_frames:4d}f ({m.receptive_field_frames*256/16000:.2f}s) "
              f"out={m.output_shape}")
