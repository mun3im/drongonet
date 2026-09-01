"""
Reproduce TailorNet's reported INT8 test accuracy for a given seed's run
by re-running the saved INT8 TFLite model against the dataset's test split.

Example:
    python eval_tailornet_seed7.py \\
        --flat_dir /path/to/mygardenbird16khz \\
        --splits_csv /path/to/splits_mip_80_10_10.csv \\
        --result_dir ../results_tailornet/tailornet_12sp_epi128_drop0.2_mixup0.2_rand42
"""
import sys, os, json, argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from tailornet import load_split

p = argparse.ArgumentParser()
p.add_argument("--flat_dir", required=True, help="MyGardenBird flat_dir")
p.add_argument("--splits_csv", required=True)
p.add_argument("--result_dir", required=True,
                help="run dir containing model_int8.tflite and tailornet_summary.json")
p.add_argument("--cm_out", default=None,
                help="optional path to save confusion matrix as .npz")
args = p.parse_args()

tflite_path = os.path.join(args.result_dir, "model_int8.tflite")

print("Loading test split...")
X_test, y_test, class_names = load_split(args.flat_dir, args.splits_csv, 'test')
print("class_names:", class_names)
print("n test:", len(X_test))

interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
in_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
in_scale, in_zp = in_d['quantization']
out_scale, out_zp = out_d['quantization']

preds = []
for x in X_test:
    xq = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    interp.set_tensor(in_d['index'], xq[np.newaxis, ...])
    interp.invoke()
    outq = interp.get_tensor(out_d['index'])[0]
    probs = (outq.astype(np.float32) - out_zp) * out_scale
    preds.append(np.argmax(probs))
preds = np.array(preds)

acc = accuracy_score(y_test, preds)
print(f"\nReproduced INT8 accuracy: {acc*100:.4f}%")

summary_path = os.path.join(args.result_dir, "tailornet_summary.json")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        recorded = json.load(f)
    print(f"Recorded INT8 accuracy (tailornet_summary.json): {recorded['int8_test_accuracy']}%")
    print(f"Match: {abs(acc*100 - recorded['int8_test_accuracy']) < 0.02}")

cm = confusion_matrix(y_test, preds)
if args.cm_out:
    np.savez(args.cm_out, cm=cm, class_names=np.array(class_names), y_test=y_test, preds=preds)
    print("Saved confusion matrix to", args.cm_out)
print(class_names)
print(cm)
