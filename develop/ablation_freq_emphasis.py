#!/usr/bin/env python3
"""
ablation_freq_emphasis.py — 3-seed ablation: GAP+Focal with vs without FrequencyEmphasis

Both models are identical (Conv8→MaxPool→Conv16→GAP→Dense, focal loss α=0.5 γ=2)
except variant B inserts a FrequencyEmphasis layer before the first Conv.

Architecture matches 3c (no FE) and 3d (with FE) at n_mels=16, n_fft=1024.
Seeds: 42, 100, 786.

Usage (assumes cache already built at CACHE_BASE_fft1024_m16):
    python ablation_freq_emphasis.py --use_cache
    python ablation_freq_emphasis.py --use_cache --seeds 42 100 786

Results written to:
    results/ablation_fe_s{seed}/no_fe/results_summary.txt
    results/ablation_fe_s{seed}/with_fe/results_summary.txt
    results/ablation_fe_summary.txt   ← final table
"""

import os, logging, time, argparse, platform, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tensorflow as tf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from tqdm import tqdm
from config import DATASET_PATH, RESULTS_BASE, CACHE_BASE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_mels: int = 16
    n_fft: int = 1024
    hop_length: int = 256
    target_sr: int = 16000
    target_length: int = 48000
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    lr_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-5
    early_stopping_patience: int = 15
    random_seed: int = 42
    dataset_path: str = DATASET_PATH

    @property
    def cache_dir(self):
        return f'{CACHE_BASE}_fft{self.n_fft}_m{self.n_mels}'

    def run_output_dir(self, variant: str) -> Path:
        return Path(RESULTS_BASE) / f'ablation_fe_s{self.random_seed}' / variant


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

class FrequencyEmphasis(tf.keras.layers.Layer):
    """Learnable per-bin sigmoid-gated spectral weighting (+F+1 params)."""

    def __init__(self, freq_bins: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.freq_bins = freq_bins

    def build(self, input_shape):
        self.freq_weights = self.add_weight(
            name='frequency_weights',
            shape=(1, 1, self.freq_bins, 1),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
        )
        self.scale = self.add_weight(
            name='scale',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(3.0),
            trainable=True,
        )

    def call(self, inputs, training=None):
        return inputs * tf.math.sigmoid(self.freq_weights * self.scale)

    def get_config(self):
        cfg = super().get_config()
        cfg['freq_bins'] = self.freq_bins
        return cfg


def build_no_fe(input_shape=(184, 16, 1), num_classes=2):
    """GAP + Focal, no FrequencyEmphasis (mirrors 3c architecture)."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='GAP_Focal_NoFE')


def build_with_fe(input_shape=(184, 16, 1), num_classes=2):
    """GAP + Focal + FrequencyEmphasis (mirrors 3d architecture)."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = FrequencyEmphasis(freq_bins=input_shape[1])(inputs)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='GAP_Focal_WithFE')


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_loss(gamma=2.0, alpha=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        fw = tf.pow(1 - pt, gamma)
        aw = y_true[:, 1:2] * alpha + y_true[:, 0:1] * (1 - alpha)
        return tf.reduce_mean(aw * fw * tf.reduce_sum(ce, axis=-1, keepdims=True))
    return loss


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_optimizer(lr):
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cached_mels(split: str, config: Config):
    cache_file = Path(config.cache_dir) / split / 'mels.npz'
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_file}\n"
            f"Build it first with one of the existing scripts (e.g. 3c_gap_focal_loss.py) "
            f"using n_mels={config.n_mels} n_fft={config.n_fft}, or pass --force-reprocess."
        )
    data = np.load(cache_file)
    logger.info(f"  Loaded {len(data['mels'])} cached mels for {split}")
    return data['mels'], data['labels']


def make_dataset(split: str, config: Config, augment: bool = False):
    mels, labels = load_cached_mels(split, config)
    mels = mels[..., np.newaxis]
    ds = tf.data.Dataset.from_tensor_slices((mels, labels))
    if split == 'train':
        ds = ds.shuffle(buffer_size=len(mels), seed=config.random_seed)
    ds = ds.map(
        lambda m, l: (m, tf.one_hot(l, depth=2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        def _aug(m, l):
            m = tf.clip_by_value(m + tf.random.normal(tf.shape(m), stddev=0.01), 0.0, 1.0)
            return m, l
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_variant(model, config: Config, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model.compile(
        optimizer=get_optimizer(config.learning_rate),
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', factor=config.lr_reduction_factor,
            patience=config.lr_patience, mode='max', min_lr=config.min_lr,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=config.early_stopping_patience,
            mode='max', restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / 'best_model.keras'),
            monitor='val_auc', mode='max', save_best_only=True,
        ),
    ]

    train_ds = make_dataset('train', config, augment=True)
    val_ds   = make_dataset('val',   config, augment=False)
    test_ds  = make_dataset('test',  config, augment=False)

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=config.epochs,
              callbacks=callbacks, verbose=1)
    train_time = time.time() - t0

    # Float32 evaluation
    probs, true_labels = [], []
    for x, y in test_ds:
        out = model(x, training=False)
        probs.extend(out[:, 1].numpy())
        true_labels.extend(np.argmax(y.numpy(), axis=1))
    probs = np.array(probs)
    true_labels = np.array(true_labels)
    float_auc = roc_auc_score(true_labels, probs)
    logger.info(f"  Float32 AUC: {float_auc:.4f}")

    # Save float model for TFLite conversion
    model.save(str(output_dir / 'best_model.keras'))

    # TFLite INT8 conversion
    def rep_ds():
        count = 0
        for x, _ in val_ds:
            for i in range(x.shape[0]):
                if count >= 500:
                    return
                yield [x[i:i+1]]
                count += 1

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.representative_dataset = rep_ds
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_bytes = converter.convert()
    except Exception:
        converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        converter2.representative_dataset = rep_ds
        tflite_bytes = converter2.convert()

    tflite_path = output_dir / 'model_int8.tflite'
    tflite_path.write_bytes(tflite_bytes)
    tflite_size_kb = len(tflite_bytes) / 1024

    # TFLite evaluation
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    inp_scale, inp_zp = inp_det['quantization']
    out_scale, out_zp = out_det['quantization']

    tflite_probs, tflite_labels, lat_ms = [], [], []
    for x, y in test_ds:
        x_np = x.numpy()
        y_np = y.numpy()
        if inp_scale != 0.0:
            x_q = np.round(x_np / inp_scale + inp_zp).astype(inp_det['dtype'])
        else:
            x_q = x_np.astype(inp_det['dtype'])
        for i in range(x_np.shape[0]):
            t = time.perf_counter()
            interp.set_tensor(inp_det['index'], x_q[i:i+1])
            interp.invoke()
            raw = interp.get_tensor(out_det['index'])
            lat_ms.append((time.perf_counter() - t) * 1000)
            if out_scale != 0.0:
                p = (raw.astype(np.float32) - out_zp) * out_scale
            else:
                p = raw.astype(np.float32)
            tflite_probs.append(float(p[0, 1]))
            tflite_labels.append(int(np.argmax(y_np[i])))

    tflite_probs = np.array(tflite_probs)
    tflite_labels = np.array(tflite_labels)
    tflite_auc = roc_auc_score(tflite_labels, tflite_probs)
    avg_lat = float(np.mean(lat_ms))

    # Save summary
    summary = (
        f"Float32 AUC:    {float_auc:.4f}\n"
        f"TFLite AUC:     {tflite_auc:.4f}\n"
        f"AUC degradation:{float_auc - tflite_auc:.4f}\n"
        f"Latency:        {avg_lat:.3f} ms/sample\n"
        f"Size:           {tflite_size_kb:.2f} KB\n"
        f"Params:         {model.count_params()}\n"
        f"Train time:     {train_time/60:.1f} min\n"
    )
    (output_dir / 'results_summary.txt').write_text(summary)
    logger.info(summary)

    return {'float_auc': float_auc, 'tflite_auc': tflite_auc,
            'latency_ms': avg_lat, 'size_kb': tflite_size_kb,
            'params': model.count_params()}


# ---------------------------------------------------------------------------
# Cache builder (only needed when --use_cache is not passed)
# ---------------------------------------------------------------------------

def _build_cache(config: Config):
    """Build mel spectrogram cache from raw audio files."""
    import pickle

    cache_root = Path(config.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    info_path = cache_root / 'cache_info.pkl'
    if info_path.exists():
        logger.info(f"Cache already exists at {cache_root}, skipping build.")
        return

    def _load_split_files(split):
        pos_dir = Path(config.dataset_path) / 'positive'
        neg_dir = Path(config.dataset_path) / 'negative'
        all_pos = sorted(pos_dir.rglob('*.wav'))
        all_neg = sorted(neg_dir.rglob('*.wav'))
        rng = random.Random(42)
        rng.shuffle(all_pos)
        rng.shuffle(all_neg)
        splits = {}
        for files, label in [(all_pos, 1), (all_neg, 0)]:
            n = len(files)
            splits.setdefault('train', []).extend([(f, label) for f in files[:int(n*0.8)]])
            splits.setdefault('val',   []).extend([(f, label) for f in files[int(n*0.8):int(n*0.9)]])
            splits.setdefault('test',  []).extend([(f, label) for f in files[int(n*0.9):]])
        return splits[split]

    cache_info = {}
    for split in ['train', 'val', 'test']:
        split_dir = cache_root / split
        split_dir.mkdir(exist_ok=True)
        pairs = _load_split_files(split)
        mels, lbls = [], []
        for fpath, label in tqdm(pairs, desc=f'Cache {split}'):
            try:
                wav, sr = librosa.load(str(fpath), sr=None)
                if sr != config.target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=config.target_sr)
                wav = wav[:config.target_length]
                if len(wav) < config.target_length:
                    wav = np.pad(wav, (0, config.target_length - len(wav)))
                ms = librosa.feature.melspectrogram(
                    y=wav, sr=config.target_sr, n_fft=config.n_fft,
                    hop_length=config.hop_length, n_mels=config.n_mels,
                    fmin=0.0, fmax=config.target_sr / 2.0, center=False,
                )
                ms = librosa.power_to_db(ms, ref=np.max).T
                ms = ms[:184, :] if ms.shape[0] > 184 else np.pad(ms, ((0, 184 - ms.shape[0]), (0, 0)))
                ms = (ms - ms.min()) / (ms.max() - ms.min() + 1e-8)
                mels.append(ms.astype(np.float32))
                lbls.append(label)
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")
        np.savez_compressed(split_dir / 'mels.npz', mels=np.array(mels), labels=np.array(lbls))
        cache_info[split] = {'n_samples': len(mels)}
    pickle.dump(cache_info, open(info_path, 'wb'))
    logger.info(f"Cache built at {cache_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='3-seed FE ablation. Reads existing results4arxiv/3c+3d runs if present.',
    )
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 786])
    p.add_argument('--n_mels', type=int, default=16)
    p.add_argument('--n_fft', type=int, default=1024)
    p.add_argument('--use_cache', action='store_true',
                   help='Load mel spectrograms from cache (add when Evo drive is mounted)')
    p.add_argument('--force_cpu', action='store_true')
    p.add_argument('--skip_existing', action='store_true', default=True,
                   help='Skip a variant/seed if results_summary.txt already exists (default: True)')
    p.add_argument('--results_base_alt', type=str, default='results4arxiv',
                   help='Alternate results base to check for pre-existing 3c/3d runs '
                        '(default: results4arxiv)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config_base = Config(n_mels=args.n_mels, n_fft=args.n_fft)
    logger.info(f"Cache dir: {config_base.cache_dir}")
    logger.info(f"Results base: {RESULTS_BASE}")

    all_results = {}  # (seed, variant) -> metrics dict

    # Map variant names to the legacy 3c/3d folder prefixes so we can reuse existing runs
    legacy_prefix = {'no_fe': '3c_gap_focal_loss', 'with_fe': '3d_gap_freq_emphasis'}

    def _read_summary(path: Path) -> dict:
        """Parse a results_summary.txt and return a metrics dict.

        Handles two historical formats:
          - 3c style: "Float32 Model AUC: 0.9808"
          - 3d style: "Float32 Model:\n  AUC: 0.9815"
        """
        row = {}
        lines = path.read_text().splitlines()
        in_float_section = False
        in_tflite_section = False
        for line in lines:
            stripped = line.strip()
            # Section headers
            if 'Float32' in line and ('Model' in line or 'AUC' in line):
                in_float_section = True
                in_tflite_section = False
            elif 'TFLite' in line and 'Model' in line:
                in_float_section = False
                in_tflite_section = True
            elif line.startswith('Timing') or (line and not line[0].isspace() and ':' not in line[:20]):
                in_float_section = False
                in_tflite_section = False

            # Parse AUC from "Float32 Model AUC: 0.9808" (3c format)
            if 'Float32 Model AUC:' in line:
                try:
                    row['float_auc'] = float(line.split(':')[-1].strip())
                except ValueError:
                    pass

            # Parse AUC from "  AUC: 0.9815" in the right section (3d format)
            if stripped.startswith('AUC:') and 'Degradation' not in line:
                try:
                    val = float(stripped.split(':')[-1].strip())
                    if in_float_section and 'float_auc' not in row:
                        row['float_auc'] = val
                    elif in_tflite_section and 'tflite_auc' not in row:
                        row['tflite_auc'] = val
                except ValueError:
                    pass

            if 'Inference Time' in line or ('Latency' in line and 'ms' in line):
                try:
                    row['latency_ms'] = float(stripped.split()[-2])
                except (ValueError, IndexError):
                    pass
            if 'Model Size' in line or ('Size' in line and 'KB' in line):
                try:
                    row['size_kb'] = float(stripped.split()[-2])
                except (ValueError, IndexError):
                    pass
            if stripped.startswith('(Experimental) Total Params') or stripped.startswith('Params'):
                try:
                    row['params'] = int(stripped.split()[-1].replace(',', ''))
                except (ValueError, IndexError):
                    pass
        return row

    for seed in args.seeds:
        for variant_name, build_fn in [('no_fe', build_no_fe), ('with_fe', build_with_fe)]:
            config = Config(n_mels=args.n_mels, n_fft=args.n_fft, random_seed=seed)
            output_dir = config.run_output_dir(variant_name)

            # 1. Check canonical output dir
            existing_summary = output_dir / 'results_summary.txt'
            # 2. Check legacy results4arxiv 3c/3d folders
            legacy_dir = (Path(args.results_base_alt) /
                          f"{legacy_prefix[variant_name]}_fft{args.n_fft}_m{args.n_mels}_s{seed}")
            legacy_summary = legacy_dir / 'results_summary.txt'

            if args.skip_existing and existing_summary.exists():
                logger.info(f"Reusing {variant_name} seed={seed} from {output_dir}")
                all_results[(seed, variant_name)] = _read_summary(existing_summary)
                continue

            if args.skip_existing and legacy_summary.exists():
                logger.info(f"Reusing {variant_name} seed={seed} from legacy {legacy_dir}")
                all_results[(seed, variant_name)] = _read_summary(legacy_summary)
                continue

            logger.info('=' * 60)
            logger.info(f"Variant: {variant_name}  |  Seed: {seed}")
            logger.info('=' * 60)

            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = build_fn(input_shape=(184, config.n_mels, 1))
            model.summary(print_fn=lambda x: logger.info(x))

            if not args.use_cache:
                logger.warning(
                    "No --use_cache flag. Will preprocess audio files via librosa. "
                    "This is slow (~30 min). Add --use_cache if the cache already exists."
                )
                _build_cache(config)

            result = train_variant(model, config, output_dir)
            all_results[(seed, variant_name)] = result

            tf.keras.backend.clear_session()

    # Write summary table
    summary_path = Path(RESULTS_BASE) / 'ablation_fe_summary.txt'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        'FrequencyEmphasis Ablation — GAP+Focal, n_mels=16, n_fft=1024',
        '=' * 70,
        f"{'Seed':>6}  {'Variant':>10}  {'Float AUC':>10}  {'TFLite AUC':>10}  "
        f"{'Δ AUC':>8}  {'Params':>7}  {'KB':>6}  {'ms':>6}",
        '-' * 70,
    ]

    for seed in args.seeds:
        for variant_name in ['no_fe', 'with_fe']:
            r = all_results.get((seed, variant_name))
            if r is None:
                lines.append(f"{seed:>6}  {variant_name:>10}  {'MISSING':>10}")
                continue
            delta = r.get('tflite_auc', 0) - r.get('float_auc', 0)
            lines.append(
                f"{seed:>6}  {variant_name:>10}  {r.get('float_auc', 0):>10.4f}  "
                f"{r.get('tflite_auc', 0):>10.4f}  {delta:>+8.4f}  "
                f"{r.get('params', 0):>7}  {r.get('size_kb', 0):>6.2f}  "
                f"{r.get('latency_ms', 0):>6.3f}"
            )

    lines.append('-' * 70)

    # Per-variant mean ± std across seeds (float AUC)
    for variant_name in ['no_fe', 'with_fe']:
        aucs = [all_results[(s, variant_name)]['float_auc']
                for s in args.seeds if (s, variant_name) in all_results]
        if aucs:
            lines.append(
                f"{'mean':>6}  {variant_name:>10}  {np.mean(aucs):>10.4f}  "
                f"{'(±' + f'{np.std(aucs):.4f}' + ')':>10}"
            )

    lines.append('')
    lines.append(f"Delta (with_fe - no_fe) per seed:")
    for seed in args.seeds:
        nf = all_results.get((seed, 'no_fe'), {}).get('float_auc')
        wf = all_results.get((seed, 'with_fe'), {}).get('float_auc')
        if nf and wf:
            lines.append(f"  seed={seed}: {wf - nf:+.4f}")

    summary_text = '\n'.join(lines) + '\n'
    summary_path.write_text(summary_text)
    print('\n' + summary_text)
    logger.info(f"Summary written to {summary_path}")


if __name__ == '__main__':
    main()
