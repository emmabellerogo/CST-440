import tensorflow as tf
import numpy as np
import pandas as pd

def convert_keras_to_tflite():
    """Convert saved Keras model to TensorFlow Lite format"""
    
    # Load the trained Keras model
    print("Loading Keras model from 'trig_model.keras'...")
    model = tf.keras.models.load_model('trig_model.keras')
    
    # Create converter
    print("Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Option 1: Basic conversion (no quantization)
    print("\nConverting to basic TFLite format...")
    tflite_model = converter.convert()
    
    # Save basic TFLite model
    with open('trig_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"✓ Saved as 'trig_model.tflite' ({size_kb:.2f} KB)")
    
    # Option 2: Quantized version (smaller size, for embedded devices)
    print("\nCreating quantized version for embedded deployment...")
    
    try:
        # Load training data for quantization representative dataset
        df = pd.read_csv('data.csv')
        x_raw = df['x'].values.astype(np.float32).reshape(-1, 1)
        x_train = x_raw / (2 * np.pi)  # Normalize to [0, 1]
        
        # Prepare representative dataset - use a list of samples
        representative_data = [x_train[i:i+1] for i in range(min(len(x_train), 30))]
        
        def representative_dataset():
            """Generator function for representative data (required for quantization)"""
            for sample in representative_data:
                yield [sample.astype(np.float32)]
        
        # Create quantized converter with all settings
        converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quant.representative_data = representative_dataset
        converter_quant.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter_quant.inference_input_type = tf.int8
        converter_quant.inference_output_type = tf.int8
        
        tflite_quantized = converter_quant.convert()
    except Exception as e:
        print(f"⚠ Full quantization failed: {e}")
        print("Creating dynamic range quantized version instead...")
        
        # Fallback: Dynamic range quantization (no representative dataset needed)
        converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized = converter_quant.convert()
    
    # Save quantized TFLite model
    with open('trig_model_quantized.tflite', 'wb') as f:
        f.write(tflite_quantized)
    
    size_quant_kb = len(tflite_quantized) / 1024
    print(f"✓ Saved as 'trig_model_quantized.tflite' ({size_quant_kb:.2f} KB)")
    
    # Print comparison
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Original Keras model:     trig_model.keras")
    print(f"Basic TFLite:             trig_model.tflite ({size_kb:.2f} KB)")
    print(f"Quantized TFLite:         trig_model_quantized.tflite ({size_quant_kb:.2f} KB)")
    print(f"Reduction:                {((size_kb - size_quant_kb) / size_kb * 100):.1f}%")
    print("\nRecommended for Arduino Nano: Use the quantized version (.tflite)")
    print("="*50)

if __name__ == "__main__":
    convert_keras_to_tflite()
