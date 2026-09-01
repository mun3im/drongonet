"""
Retrain TailorNet for one seed while logging a combined, continuously
numbered warmup+finetune epoch history to CSV (tailornet.py's own run
logs the two stages separately; this variant is for plotting one
unbroken training-history curve, e.g. for a paper figure).

Example:
    python retrain_tailornet_seed7_history.py \\
        --flat_dir /path/to/mygardenbird16khz \\
        --splits_csv /path/to/splits_mip_80_10_10.csv \\
        --output_dir ./tailornet_seed7_history_rerun \\
        --random_seed 7
"""
import sys, os, csv, argparse
import numpy as np
import tensorflow as tf
import tf_keras as keras

sys.path.insert(0, os.path.dirname(__file__))
from tailornet import load_split, create_tailornet, MixupDataGenerator

p = argparse.ArgumentParser()
p.add_argument("--flat_dir", required=True, help="MyGardenBird flat_dir")
p.add_argument("--splits_csv", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--random_seed", type=int, default=7)
p.add_argument("--warmup_epochs", type=int, default=70)
p.add_argument("--finetune_epochs", type=int, default=20)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--warmup_lr", type=float, default=1e-3)
p.add_argument("--finetune_lr", type=float, default=1e-5)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--epilogue_filters", type=int, default=128)
p.add_argument("--mixup", type=float, default=0.2, help="mixup alpha, 0 to disable")
args = p.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

np.random.seed(args.random_seed)
tf.random.set_seed(args.random_seed)

print("Loading train split...")
X_train, y_train, class_names = load_split(args.flat_dir, args.splits_csv, 'train')
print("Loading val split...")
X_val, y_val, _ = load_split(args.flat_dir, args.splits_csv, 'val')
print("Loading test split...")
X_test, y_test, _ = load_split(args.flat_dir, args.splits_csv, 'test')

num_classes = len(class_names)
print(f"{num_classes} classes, train {len(X_train)} / val {len(X_val)} / test {len(X_test)}")

model = create_tailornet(num_classes, epilogue_filters=args.epilogue_filters, dropout=args.dropout)
print(f"Params: {model.count_params():,}")

train_gen = MixupDataGenerator(X_train, y_train, args.batch_size, alpha=args.mixup, num_classes=num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)

csv_path = os.path.join(args.output_dir, "training_history.csv")

# ---- Stage 1: warmup ----
print(f"STAGE 1: WARMUP ({args.warmup_epochs} epochs)")
model.compile(optimizer=keras.optimizers.Adam(args.warmup_lr),
              loss='categorical_crossentropy', metrics=['accuracy'])
cb1 = [
    keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, "warmup_best.weights.h5"),
                                     save_best_only=True, save_weights_only=True,
                                     monitor='val_accuracy'),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    keras.callbacks.CSVLogger(os.path.join(args.output_dir, "history_stage1.csv")),
]
h1 = model.fit(train_gen, validation_data=(X_val, y_val_cat),
               epochs=args.warmup_epochs, callbacks=cb1, verbose=2)

# ---- Stage 2: finetune ----
print(f"STAGE 2: FINE-TUNING ({args.finetune_epochs} epochs)")
model.compile(optimizer=keras.optimizers.Adam(args.finetune_lr),
              loss='categorical_crossentropy', metrics=['accuracy'])
cb2 = [
    keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, "finetune_best.weights.h5"),
                                     save_best_only=True, save_weights_only=True,
                                     monitor='val_accuracy'),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                   restore_best_weights=True),
    keras.callbacks.CSVLogger(os.path.join(args.output_dir, "history_stage2.csv")),
]
h2 = model.fit(train_gen, validation_data=(X_val, y_val_cat),
               epochs=args.finetune_epochs, callbacks=cb2, verbose=2)

# ---- Combine histories with continuous epoch numbering ----
rows = []
for i, (acc, val_acc, loss, val_loss) in enumerate(zip(
        h1.history['accuracy'], h1.history['val_accuracy'],
        h1.history['loss'], h1.history['val_loss'])):
    rows.append({'epoch': i, 'stage': 'warmup', 'accuracy': acc, 'val_accuracy': val_acc,
                 'loss': loss, 'val_loss': val_loss})
offset = len(h1.history['accuracy'])
for i, (acc, val_acc, loss, val_loss) in enumerate(zip(
        h2.history['accuracy'], h2.history['val_accuracy'],
        h2.history['loss'], h2.history['val_loss'])):
    rows.append({'epoch': offset + i, 'stage': 'finetune', 'accuracy': acc, 'val_accuracy': val_acc,
                 'loss': loss, 'val_loss': val_loss})

with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['epoch', 'stage', 'accuracy', 'val_accuracy', 'loss', 'val_loss'])
    w.writeheader()
    w.writerows(rows)
print("Saved combined history to", csv_path)

# ---- Sanity check against tailornet.py's own recorded run for this seed ----
from sklearn.metrics import accuracy_score
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\nFP32 test accuracy this rerun: {acc*100:.2f}% "
      f"(compare against results_tailornet/*_rand{args.random_seed}/tailornet_summary.json)")
