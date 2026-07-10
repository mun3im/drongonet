"""bulbul_arch.py — faithful reimplementation of the bulbul CNN (Grill & Schlüter 2017,
EUSIPCO, Table I), adapted to our cross-corpus protocol's log-mel features.

Table I (their 1000x80 input, 373,169 params):
    Input 1x1000x80 -> BatchNorm (per-frequency-band normalization, axis=freq)
    Conv(3x3,16) -> Pool(3x3) -> Conv(3x3,16) -> Pool(3x3)
    Conv(3x1,16) -> Pool(3x1) -> Conv(3x1,16) -> Pool(3x1)
    Dense(256) -> Dense(32) -> Dense(1)
    leaky-ReLU max(x, x/100) everywhere except the (sigmoid) output;
    50% dropout on the inputs of the three dense layers.

Input: the genuine full 10 s clip as one contiguous 1000x80 log-mel spectrogram
(hop=160 = 10 ms/frame at 16 kHz -> ~1000 frames), i.e. bulbul's literal Table-I input. At this
size the architecture reproduces bulbul's capacity to the parameter (~373,202 vs 373,169).

Deviations (required only to share the loss/feature convention with DrongoNet, not the capacity):
  * 2-unit softmax head + focal loss, matching DrongoNet's head so the loss/output convention
    is identical across all compared models (bulbul's original sigmoid+BCE would be a confound;
    this accounts for the +33 params vs their 373,169).
  * No explicit mean-over-time subtraction (bulbul's colored-noise step); the per-frequency
    input BatchNorm covers feature normalization, consistent with how our mels are consumed.
"""
import tensorflow as tf


def build_bulbul(input_shape=(1000, 80, 1), num_classes=2):
    L = tf.keras.layers
    inputs = L.Input(shape=input_shape)

    # per-frequency-band normalization (bulbul applies BN prior to the first layer)
    x = L.BatchNormalization(axis=2)(inputs)

    # wide-receptive-field conv stack (valid padding, floor pooling) -> 16 feature maps
    x = L.Conv2D(16, (3, 3), padding='valid')(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.MaxPooling2D((3, 3))(x)
    x = L.Conv2D(16, (3, 3), padding='valid')(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.MaxPooling2D((3, 3))(x)
    x = L.Conv2D(16, (3, 1), padding='valid')(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.MaxPooling2D((3, 1))(x)
    x = L.Conv2D(16, (3, 1), padding='valid')(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.MaxPooling2D((3, 1))(x)

    x = L.Flatten()(x)

    # three dense layers, each with 50% dropout on its input
    x = L.Dropout(0.5)(x)
    x = L.Dense(256)(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(32)(x)
    x = L.LeakyReLU(alpha=0.01)(x)
    x = L.Dropout(0.5)(x)
    outputs = L.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs, name='bulbul')


if __name__ == '__main__':
    m = build_bulbul()
    m.summary()
    print('total params:', m.count_params())
