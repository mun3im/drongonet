"""
Render a confusion-matrix figure from a .npz saved by eval_tailornet_seed7.py.

Example:
    python plot_tailornet_cm.py --cm_npz tailornet_seed7_12sp_cm.npz \\
        --out tailornet_seed7_12sp_confusion.pdf \\
        --title "TailorNet (12sp, seed 7, INT8) - test confusion matrix"
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--cm_npz", required=True, help="output of eval_tailornet_seed7.py --cm_out")
p.add_argument("--out", required=True, help="output figure path (.pdf/.png)")
p.add_argument("--title", default="TailorNet - test confusion matrix")
p.add_argument("--abbrev_json", default=None,
                help="optional JSON file mapping full class names to short axis labels")
args = p.parse_args()

d = np.load(args.cm_npz, allow_pickle=True)
cm = d["cm"]
class_names = list(d["class_names"])

if args.abbrev_json:
    import json
    with open(args.abbrev_json) as f:
        abbrev = json.load(f)
    labels = [abbrev.get(c, c) for c in class_names]
else:
    labels = class_names

fig, ax = plt.subplots(figsize=(6.2, 5.6))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

n = len(class_names)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("Predicted", fontsize=9)
ax.set_ylabel("True", fontsize=9)

thresh = cm.max() / 2.0
for i in range(n):
    for j in range(n):
        v = cm[i, j]
        if v == 0:
            continue
        ax.text(j, i, str(v), ha="center", va="center",
                fontsize=7, color="white" if v > thresh else "black")

ax.set_title(args.title, fontsize=9)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
