"""
benchmark_archs.py — the published architectures we compare against, reimplemented
so every model is trained and evaluated under one identical protocol.

Follows the approach drongonet used in its own bulbul_arch.py: run *their* architecture
in *our* pipeline and cite the AUC we measure, NOT the paper's published figure. The
paper's 0.887 came from 22.05 kHz features, mean subtraction, a 5-fold ensemble and its
own augmentation recipe; quoting it beside our numbers would compare pipelines, not
architectures.

Contents:
  build_sparrow()          Grill & Schluter 2017 Table II (the local model, 309,843 params)
  build_bulbul()           Grill & Schluter 2017 Table I  (the global model, 373,169 params)
  build_drongonet_edge()   drongonet's 80-mel SBC variant (25,890 params)

Frontend deviation, stated plainly: all three run on OUR features (16 kHz, n_fft=1024,
hop=256, 80 mels, 622 frames for a 10s clip), not their native ones (sparrow/bulbul used
22.05 kHz at 70 fps). Architecture shape is preserved; the frames/params differ where the
input geometry forces it. Each builder prints its own param count so any divergence from
the published figure is visible rather than assumed.
"""

import tensorflow as tf

L = tf.keras.layers


def build_sparrow(input_shape=(622, 80, 1), num_classes=2, dropout=0.5):
    """
    Table II, the 'local' submission: a short receptive field (103 frames) slid over the
    clip, ending in a GlobalMax over local predictions. This is the direct ancestor of
    SparrowNet's topology — the comparison of interest is what that same idea costs at
    309k params versus ~2k.

    Table II layer shapes (their 701x80 input):
        Conv3x3 32 -> Conv3x3 32 -> Pool3x3 -> Conv3x3 32 -> Conv3x3 32
        -> Conv3x19 64 -> Pool3x3 -> Conv9x1 256 -> Conv1x1 64 -> Conv1x1 1 -> GlobalMax
    Batch norm on all layers, leaky ReLU max(x, x/100) except the final sigmoid.
    """
    inputs = L.Input(shape=input_shape)
    x = L.BatchNormalization(axis=2)(inputs)

    def conv_bn(x, filters, kernel, name):
        x = L.Conv2D(filters, kernel, padding="valid", name=name)(x)
        x = L.BatchNormalization()(x)
        return L.LeakyReLU(alpha=0.01)(x)

    x = conv_bn(x, 32, (3, 3), "conv1")
    x = conv_bn(x, 32, (3, 3), "conv2")
    x = L.MaxPooling2D((3, 3))(x)
    x = conv_bn(x, 32, (3, 3), "conv3")
    x = conv_bn(x, 32, (3, 3), "conv4")
    x = conv_bn(x, 64, (3, 19), "conv5")     # collapses frequency: 21 -> 3
    x = L.MaxPooling2D((3, 3))(x)
    x = conv_bn(x, 256, (9, 1), "conv6")     # the 103-frame temporal receptive field
    x = L.Dropout(dropout)(x)
    x = conv_bn(x, 64, (1, 1), "conv7")
    x = L.Dropout(dropout)(x)
    x = L.Conv2D(num_classes, (1, 1), padding="valid", name="local_logits")(x)

    # global max over the local predictions (their standard MI assumption)
    x = L.Reshape((-1, num_classes))(x)
    x = L.GlobalMaxPooling1D()(x)
    outputs = L.Softmax()(x)
    return tf.keras.Model(inputs, outputs, name="sparrow")


def build_bulbul(input_shape=(622, 80, 1), num_classes=2, dropout=0.5):
    """Table I, the 'global' submission: one wide receptive field over the whole clip
    into a dense head. Ported from drongonet's bulbul_arch.py so the two projects'
    bulbul numbers stay comparable."""
    inputs = L.Input(shape=input_shape)
    x = L.BatchNormalization(axis=2)(inputs)

    for filters, kernel, pool in [(16, (3, 3), (3, 3)), (16, (3, 3), (3, 3)),
                                  (16, (3, 1), (3, 1)), (16, (3, 1), (3, 1))]:
        x = L.Conv2D(filters, kernel, padding="valid")(x)
        x = L.LeakyReLU(alpha=0.01)(x)
        x = L.MaxPooling2D(pool)(x)

    x = L.Flatten()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256)(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(32)(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.Dropout(dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="bulbul")


def build_drongonet_edge(input_shape=(622, 80, 1), num_classes=2, dropout=0.3):
    """drongonet-edge: the 80-mel SBC variant (25,890 params at its native 184x80 input).
    No FrequencyEmphasis (edge omits it), conv stack -> GAP -> Dense."""
    inputs = L.Input(shape=input_shape)
    x = L.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = L.MaxPooling2D((2, 2))(x)
    x = L.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="drongonet_edge")


BUILDERS = {
    "sparrow": build_sparrow,
    "bulbul": build_bulbul,
    "drongonet_edge": build_drongonet_edge,
}

PUBLISHED_PARAMS = {"sparrow": 309843, "bulbul": 373169, "drongonet_edge": 25890}


if __name__ == "__main__":
    for name, builder in BUILDERS.items():
        m = builder()
        pub = PUBLISHED_PARAMS[name]
        got = m.count_params()
        print(f"{name:16s} params={got:8d}  published={pub:8d}  ratio={got/pub:.3f}  "
              f"out={m.output_shape}")
