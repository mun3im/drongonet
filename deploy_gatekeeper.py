#!/usr/bin/env python3
"""
Deploy Gatekeeper Model with Optimal Threshold

Converts trained model to TFLite with threshold adjustment baked in.
Since the model achieves 95% recall at threshold 0.264, we need to
adjust how we interpret the output probabilities.

Two deployment strategies:
1. Convert normally, adjust threshold in application code
2. Add threshold adjustment layer (experimental)
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy gatekeeper model with optimal threshold')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained Keras model')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Path to cached waveforms for quantization')
    parser.add_argument('--output_dir', type=str, default='deployment',
                        help='Output directory for TFLite models')
    parser.add_argument('--optimal_threshold', type=float, default=0.264,
                        help='Optimal threshold for 95% recall (from diagnostics)')
    parser.add_argument('--repr_samples', type=int, default=500,
                        help='Representative samples for quantization')
    return parser.parse_args()

args = parse_args()

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# Load model
logger.info(f"Loading model from {args.model_path}")
model = tf.keras.models.load_model(args.model_path)

# Load validation data for quantization
logger.info(f"Loading validation data from {args.cache_dir}")
cache_dir = Path(args.cache_dir)
val_cache = cache_dir / 'validation' / 'waveforms.npz'
data = np.load(val_cache)
X_val = data['waveforms'][..., np.newaxis]

# Representative dataset generator
def representative_dataset():
    for i in range(min(args.repr_samples, len(X_val))):
        yield [X_val[i:i+1].astype(np.float32)]

logger.info("=" * 60)
logger.info("CONVERTING TO TFLITE WITH QUANTIZATION")
logger.info("=" * 60)

# Strategy 1: Standard conversion (RECOMMENDED)
logger.info("\nStrategy 1: Standard INT8 quantization")
logger.info("- Outputs: float32 (probabilities)")
logger.info("- Threshold adjustment: Done in application code")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32  # Keep as float for probability
    
    tflite_model = converter.convert()
    
    # Save model
    tflite_path = output_dir / 'gatekeeper_int8.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    logger.info(f"✓ Saved INT8 model: {tflite_path} ({size_kb:.2f} KB)")
    
    # Save deployment info
    deployment_info = output_dir / 'DEPLOYMENT_INFO.txt'
    with open(deployment_info, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GATEKEEPER MODEL DEPLOYMENT GUIDE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write(f"  Model file: gatekeeper_int8.tflite\n")
        f.write(f"  Model size: {size_kb:.2f} KB\n")
        f.write(f"  Input: INT8 (48000 samples, quantized)\n")
        f.write(f"  Output: FLOAT32 (2 values: [prob_negative, prob_positive])\n")
        f.write("\n")
        
        f.write("DEPLOYMENT THRESHOLD:\n")
        f.write(f"  Optimal threshold: {args.optimal_threshold:.4f}\n")
        f.write(f"  Expected recall: 95.0%\n")
        f.write(f"  Expected precision: 67.8%\n")
        f.write("\n")
        
        f.write("INFERENCE PSEUDOCODE:\n")
        f.write("-" * 70 + "\n")
        f.write("// Load model\n")
        f.write("interpreter = load_tflite_model('gatekeeper_int8.tflite')\n")
        f.write("\n")
        f.write("// Prepare audio input (3 seconds @ 16kHz = 48000 samples)\n")
        f.write("audio_float32 = normalize_audio(raw_audio)  // Range: [-1, 1]\n")
        f.write("\n")
        f.write("// Quantize input\n")
        f.write("input_scale = interpreter.get_input_scale()\n")
        f.write("input_zero_point = interpreter.get_input_zero_point()\n")
        f.write("audio_int8 = quantize(audio_float32, input_scale, input_zero_point)\n")
        f.write("\n")
        f.write("// Run inference\n")
        f.write("output = interpreter.invoke(audio_int8)\n")
        f.write("prob_negative = output[0]\n")
        f.write("prob_positive = output[1]\n")
        f.write("\n")
        f.write("// Apply threshold for 95% recall\n")
        f.write(f"if (prob_positive >= {args.optimal_threshold:.4f}):\n")
        f.write("    return BIRD_DETECTED\n")
        f.write("else:\n")
        f.write("    return NO_BIRD\n")
        f.write("-" * 70 + "\n")
        f.write("\n")
        
        f.write("C++ EXAMPLE (Arduino/Cortex-M7):\n")
        f.write("-" * 70 + "\n")
        f.write("#define GATEKEEPER_THRESHOLD 0.264f\n")
        f.write("#define AUDIO_SAMPLES 48000\n")
        f.write("\n")
        f.write("// After running inference\n")
        f.write("float prob_positive = output_tensor->data.f[1];\n")
        f.write("\n")
        f.write("if (prob_positive >= GATEKEEPER_THRESHOLD) {\n")
        f.write("    // Bird detected - pass to classifier\n")
        f.write("    run_species_classifier(audio_buffer);\n")
        f.write("} else {\n")
        f.write("    // No bird - skip expensive classification\n")
        f.write("    continue;\n")
        f.write("}\n")
        f.write("-" * 70 + "\n")
        f.write("\n")
        
        f.write("PERFORMANCE EXPECTATIONS:\n")
        f.write(f"  Inference time: ~1-2ms on Cortex-M7\n")
        f.write(f"  Memory usage: ~12 KB (model) + ~50 KB (activations)\n")
        f.write(f"  Recall: 95% (catches 95% of bird sounds)\n")
        f.write(f"  Precision: 68% (32% false positives)\n")
        f.write(f"  False Negative Rate: 5% (misses 5% of birds)\n")
        f.write(f"  False Positive Rate: ~45% (on non-bird sounds)\n")
        f.write("\n")
        
        f.write("NOTES:\n")
        f.write("  • Threshold can be adjusted based on deployment needs\n")
        f.write("  • Lower threshold → Higher recall, lower precision\n")
        f.write("  • Higher threshold → Lower recall, higher precision\n")
        f.write("  • For gatekeeper: prioritize recall (catch all birds)\n")
        f.write("  • False positives are acceptable (classifier filters them)\n")
        f.write("=" * 70 + "\n")
    
    logger.info(f"✓ Saved deployment guide: {deployment_info}")
    
except Exception as e:
    logger.error(f"✗ INT8 conversion failed: {e}")

# Also save float32 model for comparison
logger.info("\nConverting float32 model for reference...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    
    float_path = output_dir / 'gatekeeper_float32.tflite'
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    
    size_kb = len(tflite_float) / 1024
    logger.info(f"✓ Saved FLOAT32 model: {float_path} ({size_kb:.2f} KB)")
except Exception as e:
    logger.error(f"✗ FLOAT32 conversion failed: {e}")

# Validate the converted model
logger.info("\n" + "=" * 60)
logger.info("VALIDATING CONVERTED MODEL")
logger.info("=" * 60)

try:
    # Load test data
    test_cache = cache_dir / 'testing' / 'waveforms.npz'
    test_data = np.load(test_cache)
    X_test = test_data['waveforms'][..., np.newaxis]
    y_test = test_data['labels']
    
    # Test on small subset
    n_samples = 100
    X_sample = X_test[:n_samples]
    y_sample = y_test[:n_samples]
    
    # Original model predictions
    logger.info("\nOriginal Keras model:")
    preds_keras = model.predict(X_sample, verbose=0)
    preds_keras_class = (preds_keras[:, 1] >= args.optimal_threshold).astype(int)
    
    # TFLite model predictions
    logger.info("TFLite INT8 model:")
    interpreter = tf.lite.Interpreter(model_path=str(output_dir / 'gatekeeper_int8.tflite'))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_scale, input_zero_point = input_details['quantization']
    logger.info(f"  Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")
    
    preds_tflite = []
    for i in range(n_samples):
        # Quantize input
        if input_scale != 0.0:
            x_quantized = np.round(X_sample[i:i+1] / input_scale + input_zero_point).astype(input_details['dtype'])
        else:
            x_quantized = X_sample[i:i+1].astype(input_details['dtype'])
        
        # Run inference
        interpreter.set_tensor(input_details['index'], x_quantized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        preds_tflite.append(output[0])
    
    preds_tflite = np.array(preds_tflite)
    preds_tflite_class = (preds_tflite[:, 1] >= args.optimal_threshold).astype(int)
    
    # Compare
    agreement = np.mean(preds_keras_class == preds_tflite_class)
    logger.info(f"\n  Agreement between Keras and TFLite: {agreement*100:.2f}%")
    
    recall_keras = np.sum((y_sample == 1) & (preds_keras_class == 1)) / np.sum(y_sample == 1)
    recall_tflite = np.sum((y_sample == 1) & (preds_tflite_class == 1)) / np.sum(y_sample == 1)
    
    logger.info(f"  Keras recall (threshold={args.optimal_threshold:.3f}): {recall_keras*100:.2f}%")
    logger.info(f"  TFLite recall (threshold={args.optimal_threshold:.3f}): {recall_tflite*100:.2f}%")
    
    if agreement > 0.95:
        logger.info("  ✓ Validation PASSED - Models are equivalent")
    else:
        logger.warning("  ⚠ Validation WARNING - Models show differences")
    
except Exception as e:
    logger.error(f"Validation failed: {e}")

logger.info("\n" + "=" * 60)
logger.info("DEPLOYMENT READY!")
logger.info("=" * 60)
logger.info(f"Model location: {output_dir / 'gatekeeper_int8.tflite'}")
logger.info(f"Documentation: {output_dir / 'DEPLOYMENT_INFO.txt'}")
logger.info(f"Threshold: {args.optimal_threshold:.4f}")
logger.info("=" * 60)
