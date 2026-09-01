"""
Plot train/val accuracy and loss curves from a training_history.csv
produced by retrain_tailornet_seed7_history.py.

Example:
    python plot_tailornet_history.py --csv training_history.csv \\
        --out tailornet_seed7_12sp_history.pdf \\
        --title "TailorNet (12sp, seed 7) - training history"
"""
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True, help="combined training_history.csv")
p.add_argument("--out", required=True, help="output figure path (.pdf/.png)")
p.add_argument("--title", default="TailorNet - training history")
args = p.parse_args()

epochs, acc, val_acc, loss, val_loss, stage = [], [], [], [], [], []
with open(args.csv) as f:
    for row in csv.DictReader(f):
        epochs.append(int(row["epoch"]))
        acc.append(float(row["accuracy"]) * 100)
        val_acc.append(float(row["val_accuracy"]) * 100)
        loss.append(float(row["loss"]))
        val_loss.append(float(row["val_loss"]))
        stage.append(row["stage"])

# boundary between warmup and finetune stages
finetune_start = next((e for e, s in zip(epochs, stage) if s == "finetune"), None)

fig, ax1 = plt.subplots(figsize=(6.4, 4.0))

l1, = ax1.plot(epochs, acc, color="#1f6f8b", linestyle="-", linewidth=1.4, label="Train accuracy")
l2, = ax1.plot(epochs, val_acc, color="#1f6f8b", linestyle="--", linewidth=1.4, label="Val accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)", color="#1f6f8b")
ax1.tick_params(axis="y", labelcolor="#1f6f8b")
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
l3, = ax2.plot(epochs, loss, color="#c1440e", linestyle="-", linewidth=1.2, alpha=0.85, label="Train loss")
l4, = ax2.plot(epochs, val_loss, color="#c1440e", linestyle="--", linewidth=1.2, alpha=0.85, label="Val loss")
ax2.set_ylabel("Categorical cross-entropy loss", color="#c1440e")
ax2.tick_params(axis="y", labelcolor="#c1440e")
ax2.set_ylim(bottom=0)

if finetune_start is not None:
    ax1.axvline(finetune_start, color="grey", linestyle=":", linewidth=1)
    ax1.text(finetune_start + 0.5, 5, "fine-tune\nstage", fontsize=7, color="grey")

lines = [l1, l2, l3, l4]
ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8, framealpha=0.9)
ax1.set_title(args.title, fontsize=10)

fig.tight_layout()
fig.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
print(f"final epoch {epochs[-1]}: train_acc={acc[-1]:.2f}% val_acc={val_acc[-1]:.2f}% "
      f"train_loss={loss[-1]:.4f} val_loss={val_loss[-1]:.4f}")
