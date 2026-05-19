import pandas as pd

df = pd.read_csv('all_results_comparison.csv')

print('=== ALL COLUMNS ===')
print(df.columns.tolist())
print()

print('=== MICRO MODELS ===')
micro = df[df['experiment'].str.contains('micro|6a_|6b_micro', case=False, na=False)]
cols = ['experiment', 'float_auc', 'tflite_auc', 'float_accuracy', 'tflite_accuracy',
        'model_size_kb', 'inference_time_ms', 'num_parameters']
avail = [c for c in cols if c in df.columns]
print(micro[avail].to_string())
print()

print('=== EDGE FINAL ===')
edge = df[df['experiment'].str.contains('6b_edge_final', case=False, na=False)]
print(edge[avail].to_string())
print()

print('=== PHASE 1 BASELINE (n_mels sweep) ===')
p1 = df[df['experiment'].str.contains('1a_baseline2d', case=False, na=False)]
print(p1[avail].to_string())
print()

print('=== PHASE 3f (Edge ablation winner) ===')
p3f = df[df['experiment'].str.contains('3f_', case=False, na=False)]
print(p3f[avail].to_string())
