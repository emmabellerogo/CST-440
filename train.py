import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def generate_training_data(num_samples=50):
    """Generate evenly-spaced training data covering full test range [-π, π]"""
    # Generate angles from -1.1π to π to match test distribution after modulo
    # This ensures the model sees the same angle ranges during training as testing
    x_raw = np.linspace(-1.1 * np.pi, np.pi, num_samples).reshape(-1, 1).astype(np.float32)
    
    # Calculate sin and cos values
    sin_vals = np.sin(x_raw).astype(np.float32)
    cos_vals = np.cos(x_raw).astype(np.float32)
    
    print(f"Generated {num_samples} evenly-spaced training samples [-1.1π, π]")
    return x_raw, np.column_stack([sin_vals, cos_vals])

# Load training data from CSV - COMMENTED OUT, using direct generation instead
# def load_trig_data(csv_path='data.csv'):
#     """Load sin, cos data from CSV file (exclude tan due to discontinuities)"""
#     df = pd.read_csv(csv_path)
#     x_raw = df['x'].values.astype(np.float32).reshape(-1, 1)
#     x_train = x_raw / (2 * np.pi)
#     y_train = df[['sin(x)', 'cos(x)']].values.astype(np.float32)
#     print(f"Loaded {len(x_train)} samples from {csv_path}")
#     return x_train, y_train
150
# Create a small model suitable for Arduino Nano
def create_trig_model():
    """Create a lightweight neural network for sin and cos functions"""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(2, activation='tanh')  # 2 outputs: sin, cos
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(epochs=300, batch_size=4, num_samples=25):
    """Train the model on trig functions - simple approach without callbacks"""
    print("Generating training data...")
    x_raw, y_train = generate_training_data(num_samples=num_samples)
    
    # Use modulo normalization (same as inference) to teach periodicity
    # This ensures 2π wraps back to 0, helping model learn periodic behavior
    x_train = np.mod(x_raw, 2 * np.pi) / (2 * np.pi)
    
    print("Creating model...")
    model = create_trig_model()
    
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[],
        verbose=1
    )
    
    # Save model
    model.save('trig_model.keras')
    print("Model saved as 'trig_model.keras'")
    
    return model, history, x_raw, y_train

def quantize_model(model):
    """Convert model to quantized TFLite format for Arduino deployment"""
    print("\n=== Quantizing Model for Arduino ===")
    
    # Generate representative data for quantization
    x_rep = np.linspace(0, 1, 20).reshape(-1, 1).astype(np.float32)
    
    def representative_dataset():
        """Generator function for representative data"""
        for i in range(len(x_rep)):
            yield [x_rep[i:i+1].astype(np.float32)]
    
    # Create quantized TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_data = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    tflite_model = converter.convert()
    
    # Save quantized model
    with open('trig_model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"Quantized model saved as 'trig_model_quantized.tflite'")
    print(f"Size: {size_kb:.2f} KB (Arduino Nano has 32 KB flash)")
    print(f"Use this file for Arduino deployment!")
    
    return tflite_model

def evaluate_model(model):
    """Evaluate model performance on test data"""
    # Generate test data
    x_test_raw = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1).astype(np.float32)
    x_test = x_test_raw / (2 * np.pi)
    y_test = np.column_stack([np.sin(x_test_raw), np.cos(x_test_raw)]).astype(np.float32)
    
    # Make predictions
    predictions = model.predict(x_test, verbose=0)
    
    # Calculate errors for each function
    sin_error = np.mean(np.abs(predictions[:, 0] - y_test[:, 0]))
    cos_error = np.mean(np.abs(predictions[:, 1] - y_test[:, 1]))
    
    print("\n=== Model Evaluation ===")
    print(f"Mean Absolute Error - Sin: {sin_error:.6f}")
    print(f"Mean Absolute Error - Cos: {cos_error:.6f}")
    
    # Plot results
    plot_predictions(model, x_test, y_test, predictions)
    
    return predictions, y_test

def plot_predictions(model, x_test, y_test, predictions):
    """Plot actual vs predicted values for sin and cos functions"""
    # Create smooth x range from 0 to 2π
    x_smooth_raw = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
    x_smooth = x_smooth_raw / (2 * np.pi)  # Normalize like training data
    
    # Make predictions on smooth range
    pred_smooth = model.predict(x_smooth, verbose=0)
    
    # Calculate true trig function values
    y_true_sin = np.sin(x_smooth_raw)
    y_true_cos = np.cos(x_smooth_raw)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    functions = ['sin(x)', 'cos(x)']
    y_true = [y_true_sin, y_true_cos]
    
    for idx, (ax, func_name, y_func) in enumerate(zip(axes, functions, y_true)):
        ax.plot(x_smooth_raw, y_func, label='Actual', linewidth=2.5, alpha=0.8)
        ax.plot(x_smooth_raw, pred_smooth[:, idx], label='Predicted', linewidth=2.5, alpha=0.8, linestyle='--')
        ax.set_xlabel('x')
        ax.set_ylabel(func_name)
        ax.set_title(f'{func_name} - Actual vs Predicted')
        ax.set_xlim(0, 2*np.pi)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_plot.png', dpi=150, bbox_inches='tight')
    print("Graph saved as 'predictions_plot.png'")
    plt.close()

if __name__ == "__main__":
    # Use 50 naturally distributed training samples
    model, history, x_raw, y_train = train_model(epochs=400, batch_size=8, num_samples=50)
    predictions, y_test = evaluate_model(model)
    tflite_model = quantize_model(model)


