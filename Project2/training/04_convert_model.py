"""
Step 4: Convert Trained Model to TensorFlow Lite
Quantizes and converts the model for microcontroller deployment.
"""

import os
import numpy as np
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "WakeWord", "src")

KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "keyword_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "keyword_model.tflite")
TFLITE_QUANT_PATH = os.path.join(MODEL_DIR, "keyword_model_quant.tflite")
HEADER_PATH = os.path.join(OUTPUT_DIR, "keyword_model_data.h")
CPP_PATH = os.path.join(OUTPUT_DIR, "keyword_model_data.cpp")

# ============================================================
# REPRESENTATIVE DATASET FOR QUANTIZATION
# ============================================================

def representative_dataset_gen():
    """Generator for representative dataset used in quantization."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))

    # Use a subset for calibration
    num_calibration = min(200, len(X_train))
    indices = np.random.choice(len(X_train), num_calibration, replace=False)

    for idx in indices:
        sample = X_train[idx:idx+1].astype(np.float32)
        yield [sample]

# ============================================================
# CONVERSION FUNCTIONS
# ============================================================

def convert_to_tflite(model_path):
    """Convert Keras model to TensorFlow Lite (float32)."""
    print("Converting to TFLite (float32)...")

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  Saved to {TFLITE_MODEL_PATH}")
    print(f"  Size: {size_kb:.1f} KB")

    return tflite_model

def convert_to_tflite_quantized(model_path):
    """Convert Keras model to quantized TensorFlow Lite (int8)."""
    print("\nConverting to TFLite (int8 quantized)...")

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(TFLITE_QUANT_PATH, 'wb') as f:
        f.write(tflite_quant_model)

    size_kb = len(tflite_quant_model) / 1024
    print(f"  Saved to {TFLITE_QUANT_PATH}")
    print(f"  Size: {size_kb:.1f} KB")

    return tflite_quant_model

def generate_c_array(tflite_model, use_quantized=True):
    """Generate C header file from TFLite model."""
    model_path = TFLITE_QUANT_PATH if use_quantized else TFLITE_MODEL_PATH

    print(f"\nGenerating C header from {'quantized' if use_quantized else 'float32'} model...")

    with open(model_path, 'rb') as f:
        model_data = f.read()

    # Generate header file
    header_content = '''#ifndef KEYWORD_MODEL_DATA_H
#define KEYWORD_MODEL_DATA_H

// Auto-generated from TensorFlow Lite model
// Model: keyword_model_quant.tflite

extern const unsigned char keyword_model_data[];
extern const unsigned int keyword_model_data_len;

// Model input/output information
#define MODEL_INPUT_FRAMES 49
#define MODEL_INPUT_MFCC 13

#endif // KEYWORD_MODEL_DATA_H
'''

    # Generate CPP file with model data
    cpp_content = '''#include "keyword_model_data.h"

// Auto-generated from TensorFlow Lite model
// Size: {} bytes

alignas(8) const unsigned char keyword_model_data[] = {{
'''.format(len(model_data))

    # Convert bytes to C array format
    bytes_per_line = 12
    for i, byte in enumerate(model_data):
        if i % bytes_per_line == 0:
            cpp_content += '    '
        cpp_content += f'0x{byte:02x},'
        if i % bytes_per_line == bytes_per_line - 1:
            cpp_content += '\n'
        else:
            cpp_content += ' '

    cpp_content += '''
};

const unsigned int keyword_model_data_len = {};
'''.format(len(model_data))

    # Write files
    os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)

    with open(HEADER_PATH, 'w') as f:
        f.write(header_content)
    print(f"  Header saved to {HEADER_PATH}")

    with open(CPP_PATH, 'w') as f:
        f.write(cpp_content)
    print(f"  Source saved to {CPP_PATH}")

    return len(model_data)

# ============================================================
# VERIFICATION
# ============================================================

def verify_tflite_model(model_path, X_test, y_test):
    """Verify TFLite model accuracy."""
    print(f"\nVerifying model: {os.path.basename(model_path)}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    # Test accuracy
    correct = 0
    total = min(500, len(X_test))  # Test on subset for speed

    input_scale = input_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
    input_zero_point = input_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]

    for i in range(total):
        sample = X_test[i:i+1].astype(np.float32)

        # Quantize input if needed
        if input_details[0]['dtype'] == np.int8:
            sample = (sample / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        pred = np.argmax(output)
        if pred == y_test[i]:
            correct += 1

    accuracy = correct / total
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Conversion for TinyML Deployment")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"\nError: Keras model not found at {KERAS_MODEL_PATH}")
        print("Please run 03_train_model.py first!")
        exit(1)

    # Load test data for verification
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Convert to TFLite (float32)
    tflite_model = convert_to_tflite(KERAS_MODEL_PATH)

    # Convert to TFLite (int8 quantized)
    tflite_quant_model = convert_to_tflite_quantized(KERAS_MODEL_PATH)

    # Verify both models
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    verify_tflite_model(TFLITE_MODEL_PATH, X_test, y_test)
    verify_tflite_model(TFLITE_QUANT_PATH, X_test, y_test)

    # Generate C header (using quantized model)
    print("\n" + "=" * 60)
    print("Generating C Header")
    print("=" * 60)
    model_size = generate_c_array(tflite_quant_model, use_quantized=True)

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nFiles generated:")
    print(f"  - {TFLITE_MODEL_PATH} (float32)")
    print(f"  - {TFLITE_QUANT_PATH} (int8)")
    print(f"  - {HEADER_PATH}")
    print(f"  - {CPP_PATH}")
    print(f"\nQuantized model size: {model_size / 1024:.1f} KB")
    print("\nNext step: Update main.cpp and deploy to Arduino")
