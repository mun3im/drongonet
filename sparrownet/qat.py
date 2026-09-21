"""
qat.py — quantization-aware training support.

drongonet's LESSONS_LEARNT.md names post-training quantization as "the #1 recall
killer" and says QAT should be the default. The obstacle is FrequencyEmphasis:
tfmot's `quantize_model` refuses to wrap an unrecognized custom layer, so we
annotate selectively — every layer tfmot natively supports gets fake-quant nodes,
FrequencyEmphasis is left alone and is handled by the converter's own calibration.

`apply_qat(model)` returns a QAT-ready clone. Train it, then convert exactly as the
PTQ path does; the fake-quant nodes make the converter use the learned ranges.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from models import FrequencyEmphasis

quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_apply = tfmot.quantization.keras.quantize_apply
quantize_scope = tfmot.quantization.keras.quantize_scope

# Layer types tfmot can quantize out of the box in the architectures we build.
_SUPPORTED = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.Dense,
    tf.keras.layers.BatchNormalization,
)

# Left unannotated on purpose:
#   FrequencyEmphasis  custom layer, no tfmot support
#   MaxPooling/GAP/GlobalMax/Reshape/Softmax/Dropout  no weights to quantize
_SKIP_BY_NAME = ()


def _annotate(layer):
    if isinstance(layer, FrequencyEmphasis) or layer.name in _SKIP_BY_NAME:
        return layer
    if isinstance(layer, _SUPPORTED):
        return quantize_annotate_layer(layer)
    return layer


def apply_qat(model):
    """Clone `model` with fake-quant nodes on every tfmot-supported layer."""
    annotated = tf.keras.models.clone_model(model, clone_function=_annotate)
    with quantize_scope({"FrequencyEmphasis": FrequencyEmphasis}):
        return quantize_apply(annotated)


def count_quantized_layers(qat_model):
    return sum(1 for l in qat_model.layers if "quant" in l.__class__.__name__.lower())


if __name__ == "__main__":
    from models import build_sparrownet, build_drongonet_micro

    for name, builder in [("sparrownet", lambda: build_sparrownet(n_time_stages=6)),
                          ("drongonet_micro", build_drongonet_micro)]:
        base = builder()
        q = apply_qat(base)
        print(f"{name:18s} base params={base.count_params():6d} "
              f"qat params={q.count_params():6d} "
              f"quant-wrapped layers={count_quantized_layers(q)}")
        # a QAT model must still train and predict
        q.compile(optimizer="adam", loss="categorical_crossentropy")
        import numpy as np
        x = np.random.rand(2, *base.input_shape[1:]).astype("float32")
        print(f"  forward ok: {q.predict(x, verbose=0).shape}")
