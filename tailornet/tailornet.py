"""
TailorNet — a franken-net for 12-species garden-bird classification.

Recipe (as specified, verified against source before building — see the
per-component provenance below, not guessed):

  1. DATASET LOADER: MyGardenBird's flat_dir/species-subdir + CSV-split
     convention, same as MynaNet's `load_data_from_csv`
     (mynanet/develop/1o_fcn_epilogue.py). One deliberate change from that
     original: class-index assignment here is **alphabetically sorted**,
     not the original's first-appearance-while-iterating-test-split-by-
     filename order. That original ordering was an accidental artifact of
     iteration structure, not a design choice, and it was the exact root
     cause of a real bug found and fixed in the OSR study on 2026-08-14
     (0.14% closed-set accuracy from a class-index mismatch) -- a fresh
     script has no legacy-compatibility reason to keep reproducing it.

  2. MEL FRONT-END: DrongoNet-Micro's exact recipe, not MynaNet's own
     (mynanet uses 64 mels/512-pt FFT/300 frames; this uses Micro's
     16 mels/1024-pt FFT/184 frames), per explicit instruction: "use
     16x184 input... in final system, i want to reuse mel-spectrograms
     used by drongonet-micro" (drongonet/develop/6b_micro_final.py
     compute_mel_spectrogram, copied verbatim below, including its
     per-clip [0,1] min-max normalisation -- NOT MynaNet 1o's global-
     train-stats normalisation, deliberately, so the exact same tensor
     DrongoNet-Micro already computes as its gate input can be handed to
     this model's Invoke() unchanged in the final cascade, with zero
     second mel/FFT pass). fmin=100Hz, fmax=8000Hz, center=False,
     power_to_db(ref=max).

  3. BACKBONE: DrongoNet-Edge's `build_deeper_gap` conv stack, verbatim
     (drongonet/develop/6c_edge_final.py) -- Conv2D(16,3x3)+BN+ReLU+
     MaxPool2D(2x2) -> Conv2D(32,3x3)+BN+ReLU+MaxPool2D(2x2) ->
     Conv2D(64,3x3)+BN+ReLU (no pool). Edge's own head (GAP+Dense(32)+
     Dropout+Dense(2)) is discarded; only the 3-block feature extractor
     is reused, now consuming (184,16,1) instead of Edge's native
     (184,80,1). NOTE: Edge's Conv2D layers use `use_bias=True` (Keras
     default, not explicitly disabled) even though BatchNorm follows --
     redundant with BN's own beta term, but preserved here for exact
     fidelity to "use DrongoNet-Edge as the backbone" rather than a
     silently "corrected" version.

  4. FINAL STAGE: model 1o's MatchboxNet-style fully-convolutional
     epilogue (mynanet/develop/1o_fcn_epilogue.py create_mbv3_matchbox,
     epilogue section only). Verified 2026-08-15 that this is NOT a
     naming conflict with "1n/WrenNet" -- 1n's own docstring states it
     reuses "1o's MatchboxNet epilogue" verbatim as its own final stage,
     so 1o's epilogue and "wrennet's final stage" are the same
     architectural component either way; no ambiguity survived the check.
     AXIS-ORDER CORRECTION: 1o's epilogue was written for MynaNet's
     (n_mels, time, 1) input convention, where its DepthwiseConv2D(1,17)
     is time-only along the *last* spatial axis. DrongoNet's convention
     (inherited here via the backbone and mel front-end) is (time, freq, 1)
     -- axes swapped. To keep the depthwise kernel genuinely time-only
     (not frequency-only, which would be architecturally wrong and was
     never the design intent of a MatchboxNet-style long temporal
     receptive field), the kernel here is DepthwiseConv2D(17,1), not
     (1,17). Get this backwards and the kernel silently convolves along
     frequency instead of time -- verified by inspecting the actual axis
     each source file assumes, not assumed by name similarity.

Objective: species recognizer with a materially smaller resource
footprint than MynaNet 1o (120,500 params, 193.4 kB INT8) via a much
lighter backbone + much smaller input. Projected (hand-computed before
training, both stages combined): ~35.3K params -- backbone ~23.7K +
epilogue ~11.6K -- roughly 3.4x fewer parameters and ~5.5x smaller INT8
flash than 1o, NOT a full 10x on flash with this exact verbatim recipe.
The actual post-training/post-quantisation numbers are what's reported at
the end of this script's run, not this estimate -- do not cite the
projection instead of the measured figure once training completes. If a
literal 10x is required, the follow-up move is halving channel widths
(8->16->32 backbone, 64 instead of 128 epilogue projection) -- not done
here, since that would silently deviate from the specified recipe rather
than report against it honestly.

Model naming: "TailorNet" -- Common Tailorbird is one of the 12
MyGardenBird species, and the name fits this project's bird-themed
naming convention (DrongoNet, MynaNet, ARGUS).
"""

import argparse
import os
import sys
import json
import platform
from datetime import datetime

# --------------------------------------------------------------
# TENSORFLOW & KERAS ENVIRONMENT CHECK (same pattern as 1o_fcn_epilogue.py)
# --------------------------------------------------------------
print("\n" + "=" * 70)
print("ENVIRONMENT VALIDATION")
print("=" * 70)
try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras import layers

    print(f"TensorFlow version: {tf.__version__}")
    print(f"tf_keras version: {keras.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s): {gpus[0].name}")
    else:
        print("Running on CPU (no GPU detected)")
    print("Environment check PASSED")
except Exception as e:
    print(f"CRITICAL: TensorFlow environment check failed: {e}")
    sys.exit(1)
print("=" * 70)

import numpy as np
import librosa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# --------------------------------------------------------------
# CONSTANTS -- DrongoNet-Micro's exact mel spec (6b_micro_final.py),
# reused verbatim per the "reuse mel-spectrograms used by drongonet-micro"
# instruction. Do not change these without re-checking that the deployed
# gate's own config still matches -- the whole point is a shared tensor.
# --------------------------------------------------------------
TARGET_SR    = 16000
N_FFT        = 1024
HOP_LENGTH   = 256
N_MELS       = 16
MEL_FMIN     = 100.0
MEL_FMAX     = 8000.0
TIME_FRAMES  = 184          # 3s @ 16kHz, hop_length=256, center=False
INPUT_SHAPE  = (TIME_FRAMES, N_MELS, 1)   # (time, freq, 1) -- DrongoNet axis order


def compute_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Verbatim port of drongonet/develop/6b_micro_final.py's
    compute_mel_spectrogram -- same FFT config, same per-clip [0,1]
    min-max normalisation. This IS the tensor DrongoNet-Micro's own gate
    computes; TailorNet is designed to consume it unchanged."""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, center=False,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = mel_spec_db.T  # (time, freq)

    if mel_spec_db.shape[0] > TIME_FRAMES:
        mel_spec_db = mel_spec_db[:TIME_FRAMES, :]
    elif mel_spec_db.shape[0] < TIME_FRAMES:
        pad = ((0, TIME_FRAMES - mel_spec_db.shape[0]), (0, 0))
        mel_spec_db = np.pad(mel_spec_db, pad, mode='constant',
                              constant_values=mel_spec_db.min())

    mel_min, mel_max = mel_spec_db.min(), mel_spec_db.max()
    if mel_max - mel_min > 1e-6:
        mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min)
    else:
        mel_spec_db = np.zeros_like(mel_spec_db)

    return mel_spec_db.astype(np.float32)[..., np.newaxis]  # (184, 16, 1)


def load_waveform(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    target_len = TARGET_SR * 3
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


# --------------------------------------------------------------
# DATASET LOADER -- MyGardenBird flat_dir/species-subdir + CSV split,
# same convention as MynaNet's load_data_from_csv, class order
# alphabetically sorted (deliberate change from 1o's original -- see
# module docstring).
# --------------------------------------------------------------
def parse_splits_csv(csv_path):
    """Return {file_id (no extension) -> split}. Tolerates a leading
    '# ...' comment line, matches every other splits CSV in this repo."""
    m = {}
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('file_id'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                m[parts[0].strip()] = parts[1].strip()
    return m


def build_file_lookup(flat_dir):
    """filename -> (class_name, full_path), class_name from alphabetically
    sorted subdirectory listing (not file-encounter order)."""
    lookup = {}
    for class_name in sorted(os.listdir(flat_dir)):
        class_dir = os.path.join(flat_dir, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                lookup[f] = (class_name, os.path.join(class_dir, f))
    return lookup


def load_split(flat_dir, splits_csv, target_split):
    """Featurise every clip belonging to `target_split` ('train'/'val'/'test').
    Returns (X, y, class_names) where class_names[i] is the alphabetically
    sorted species name for label i."""
    splits = parse_splits_csv(splits_csv)
    lookup = build_file_lookup(flat_dir)
    class_names = sorted({c for c, _ in lookup.values()})
    cls_idx = {c: i for i, c in enumerate(class_names)}

    # splits CSV file_ids may or may not carry the .wav extension across
    # this repo's various CSVs -- handle both, matching parse conventions
    # already established elsewhere (osr_common.featurise_dir's callers).
    resolved = []
    for fn, sp in sorted(splits.items()):
        if sp != target_split:
            continue
        key = fn if fn in lookup else (fn + '.wav')
        if key in lookup:
            resolved.append(lookup[key])

    X, y = [], []
    for class_name, full_path in tqdm(resolved, desc=f"featurising {target_split}"):
        try:
            wav = load_waveform(full_path)
            X.append(compute_mel_spectrogram(wav))
            y.append(cls_idx[class_name])
        except Exception as e:
            print(f"  warn: {full_path}: {e}")

    return np.stack(X), np.array(y, dtype=np.int32), class_names


# --------------------------------------------------------------
# MODEL -- DrongoNet-Edge backbone + 1o epilogue, axis-corrected.
# --------------------------------------------------------------
def build_backbone(x):
    """Verbatim DrongoNet-Edge build_deeper_gap conv stack (Edge's own
    head discarded -- only these three blocks are reused)."""
    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def build_epilogue(x, num_classes, epilogue_filters=128, dropout=0.2):
    """1o's MatchboxNet-style FCN epilogue, axis-corrected: the depthwise
    kernel is (17,1) here (time-only along axis 0, this codebase's time
    axis), not 1o's own (1,17) (which was time-only along axis 1, under
    MynaNet's opposite (freq,time,1) convention). Same kernel WIDTH (17),
    same structure, deliberately not re-tuned for this input's different
    post-backbone time resolution -- see module docstring."""
    x = layers.DepthwiseConv2D((17, 1), padding='same',
                                use_bias=False, name='epilogue_dw')(x)
    x = layers.BatchNormalization(name='epilogue_dw_bn')(x)
    x = layers.ReLU(6., name='epilogue_dw_relu')(x)

    x = layers.Conv2D(epilogue_filters, (1, 1), padding='same',
                       use_bias=False, name='epilogue_pw')(x)
    x = layers.BatchNormalization(name='epilogue_pw_bn')(x)
    x = layers.ReLU(6., name='epilogue_pw_relu')(x)

    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dropout(dropout, name='fc_drop')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return outputs


def create_tailornet(num_classes, input_shape=INPUT_SHAPE,
                      epilogue_filters=128, dropout=0.2):
    inputs = layers.Input(shape=input_shape, name='input')
    x = build_backbone(inputs)
    outputs = build_epilogue(x, num_classes, epilogue_filters, dropout)
    return keras.Model(inputs, outputs, name="TailorNet")


# --------------------------------------------------------------
# MIXUP DATA GENERATOR (same pattern as 1o_fcn_epilogue.py)
# --------------------------------------------------------------
class MixupDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2, num_classes=12):
        self.X, self.y = X, y
        self.batch_size, self.alpha, self.num_classes = batch_size, alpha, num_classes
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        b = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, y_batch = self.X[b], self.y[b]
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            perm = np.random.permutation(len(X_batch))
            X_mixed = lam * X_batch + (1 - lam) * X_batch[perm]
            y_a = keras.utils.to_categorical(y_batch, self.num_classes)
            y_b = keras.utils.to_categorical(y_batch[perm], self.num_classes)
            return X_mixed, lam * y_a + (1 - lam) * y_b
        return X_batch, keras.utils.to_categorical(y_batch, self.num_classes)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# --------------------------------------------------------------
# TFLITE INT8 EXPORT (same pattern as every other script in this repo)
# --------------------------------------------------------------
def convert_to_tflite_int8(model, X_calib, path):
    def rep_data():
        for i in range(min(200, len(X_calib))):
            yield [X_calib[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"Saved INT8 TFLite: {path} ({size_kb:.1f} KB)")
    return tflite_model, size_kb


def evaluate_tflite(tflite_path, X_test, y_test, class_names, output_dir):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = in_d['quantization']
    out_scale, out_zp = out_d['quantization']

    preds = []
    for x in tqdm(X_test, desc="INT8 inference"):
        xq = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(in_d['index'], xq[np.newaxis, ...])
        interp.invoke()
        outq = interp.get_tensor(out_d['index'])[0]
        probs = (outq.astype(np.float32) - out_zp) * out_scale
        preds.append(np.argmax(probs))
    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, "classification_report_int8.txt"), 'w') as f:
        f.write(report)
    print(f"\nINT8 Classification Report:\n{report}")
    return acc


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flat_dir", required=True, help="12-species MyGardenBird flat_dir")
    p.add_argument("--splits_csv", required=True)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--warmup_epochs", type=int, default=70)
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup_lr", type=float, default=1e-3)
    p.add_argument("--finetune_lr", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epilogue_filters", type=int, default=128)
    p.add_argument("--mixup", type=float, default=0.2, help="mixup alpha, 0 to disable")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Species count from flat_dir directly (same approach as
    # 1o_fcn_epilogue.py's n_classes computation), BEFORE output_dir is
    # built. Added 2026-08-15 after a near-miss: 1o_fcn_epilogue.py's own
    # output-dir template never encoded species count either, and a
    # 14-species retrain with the same seed/hyperparams as an existing
    # 12-species run would have silently landed in and overwritten the
    # canonical checkpoint's directory (same split-ratio header, same
    # hyperparams, nothing in the name to tell the two runs apart). Caught
    # before it happened this session -- encoding species count here so
    # the same mistake can't recur silently for TailorNet either.
    n_classes_for_naming = len([d for d in os.listdir(args.flat_dir)
                                 if os.path.isdir(os.path.join(args.flat_dir, d))
                                 and not d.startswith('.')])
    output_dir = args.output_dir or (
        f"results_tailornet/tailornet_{n_classes_for_naming}sp_epi{args.epilogue_filters}_"
        f"drop{args.dropout}_mixup{args.mixup}_rand{args.random_seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfig:")
    print(f"  Random seed:      {args.random_seed}")
    print(f"  Warmup/Finetune:  {args.warmup_epochs}/{args.finetune_epochs} epochs")
    print(f"  Input shape:      {INPUT_SHAPE} (DrongoNet-Micro mel spec)")
    print(f"  Backbone:         DrongoNet-Edge build_deeper_gap (16->32->64ch)")
    print(f"  Epilogue:         1o MatchboxNet FCN, {args.epilogue_filters}ch projection")
    print(f"  Output dir:       {output_dir}")

    print("\nLoading train split...")
    X_train, y_train, class_names = load_split(args.flat_dir, args.splits_csv, 'train')
    print("\nLoading val split...")
    X_val, y_val, _ = load_split(args.flat_dir, args.splits_csv, 'val')
    print("\nLoading test split...")
    X_test, y_test, _ = load_split(args.flat_dir, args.splits_csv, 'test')

    num_classes = len(class_names)
    print(f"\n{num_classes} classes: {class_names}")
    print(f"Train {len(X_train)} / Val {len(X_val)} / Test {len(X_test)}")

    model = create_tailornet(num_classes, epilogue_filters=args.epilogue_filters,
                              dropout=args.dropout)
    model.summary()
    n_params = model.count_params()
    with open(os.path.join(output_dir, "model_summary.txt"), 'w') as f:
        model.summary(print_fn=lambda s: f.write(s + '\n'))
    print(f"\nTotal params: {n_params:,}  "
          f"(1o reference: 120,500 -- ratio: {120500/n_params:.2f}x fewer)")

    # ---- Stage 1: warmup ----
    print("\n" + "=" * 70)
    print(f"STAGE 1: WARMUP ({args.warmup_epochs} epochs)")
    print("=" * 70)
    model.compile(optimizer=keras.optimizers.Adam(args.warmup_lr),
                   loss='categorical_crossentropy', metrics=['accuracy'])
    train_gen = MixupDataGenerator(X_train, y_train, args.batch_size,
                                    alpha=args.mixup, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    ckpt1 = os.path.join(output_dir, "warmup_best.weights.h5")
    cb1 = [
        keras.callbacks.ModelCheckpoint(ckpt1, save_best_only=True,
                                         save_weights_only=True, monitor='val_accuracy'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ]
    model.fit(train_gen, validation_data=(X_val, y_val_cat),
              epochs=args.warmup_epochs, callbacks=cb1, verbose=1)

    # ---- Stage 2: finetune ----
    print("\n" + "=" * 70)
    print(f"STAGE 2: FINE-TUNING ({args.finetune_epochs} epochs)")
    print("=" * 70)
    model.compile(optimizer=keras.optimizers.Adam(args.finetune_lr),
                   loss='categorical_crossentropy', metrics=['accuracy'])
    ckpt2 = os.path.join(output_dir, "finetune_best.weights.h5")
    cb2 = [
        keras.callbacks.ModelCheckpoint(ckpt2, save_best_only=True,
                                         save_weights_only=True, monitor='val_accuracy'),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                       restore_best_weights=True),
    ]
    model.fit(train_gen, validation_data=(X_val, y_val_cat),
              epochs=args.finetune_epochs, callbacks=cb2, verbose=1)

    # ---- Evaluate FP32 ----
    y_pred_fp32 = np.argmax(model.predict(X_test, verbose=0), axis=1)
    fp32_acc = accuracy_score(y_test, y_pred_fp32)
    print(f"\nFP32 test accuracy: {fp32_acc*100:.2f}%")
    report_fp32 = classification_report(y_test, y_pred_fp32, target_names=class_names, digits=4)
    with open(os.path.join(output_dir, "classification_report_fp32.txt"), 'w') as f:
        f.write(report_fp32)

    keras_path = os.path.join(output_dir, "model_fp32.keras")
    model.save(keras_path)

    # ---- Export + evaluate INT8 ----
    print("\n" + "=" * 70)
    print("INT8 EXPORT + EVALUATION")
    print("=" * 70)
    tflite_path = os.path.join(output_dir, "model_int8.tflite")
    _, size_kb = convert_to_tflite_int8(model, X_train, tflite_path)
    int8_acc = evaluate_tflite(tflite_path, X_test, y_test, class_names, output_dir)

    # ---- Final report ----
    summary = {
        "n_params": int(n_params),
        "int8_size_kb": round(size_kb, 1),
        "fp32_test_accuracy": round(fp32_acc * 100, 2),
        "int8_test_accuracy": round(int8_acc * 100, 2),
        "accuracy_drop_pp": round((fp32_acc - int8_acc) * 100, 2),
        "reference_1o_params": 120500,
        "reference_1o_int8_kb": 193.4,
        "params_reduction_x": round(120500 / n_params, 2),
        "flash_reduction_x": round(193.4 / size_kb, 2),
        "random_seed": args.random_seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "tailornet_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
