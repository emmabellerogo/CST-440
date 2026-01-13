import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load training data from CSV
def load_trig_data(csv_path='data.csv'):
    """Load sin, cos data from CSV file (exclude tan due to discontinuities)"""
    df = pd.read_csv(csv_path)
    
    # Normalize x to [0, 1] range for better neural network training
    x_raw = df['x'].values.astype(np.float32).reshape(-1, 1)
    x_train = x_raw / (2 * np.pi)  # Normalize to [0, 1] assuming x in [0, 2π]
    y_train = df[['sin(x)', 'cos(x)']].values.astype(np.float32)
    
    print(f"Loaded {len(x_train)} samples from {csv_path}")
    return x_train, y_train

# Create a small model suitable for Arduino Nano
def create_trig_model():
    """Create a lightweight neural network for sin and cos functions"""
    model = keras.Sequential([
        keras.layers.Dense(48, activation='relu', input_shape=(1,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(48, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(2, activation='linear')  # 2 outputs: sin, cos
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(epochs=300, batch_size=8):
    """Train the model on trig functions with learning rate scheduling and early stopping"""
    print("Loading training data...")
    x_train, y_train = load_trig_data('data.csv')
    
    print("Creating model...")
    model = create_trig_model()
    
    # Learning rate scheduling: reduce LR when validation loss plateaus
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    # Early stopping: stop if validation loss doesn't improve
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )
    
    # Save model
    model.save('trig_model.keras')
    print("Model saved as 'trig_model.keras'")
    
    return model, history

def quantize_model(model):
    """Convert model to quantized TFLite format for Arduino deployment"""
    print("\n=== Quantizing Model for Arduino ===")
    
    # Load training data to use as representative dataset for quantization
    x_train, _ = load_trig_data('data.csv')
    
    def representative_dataset():
        """Generator function for representative data (required for quantization)"""
        for i in range(min(len(x_train), 30)):  # Use subset of data
            yield [x_train[i:i+1]]
    
    # Create quantized TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_data = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
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
    x_test, y_test = load_trig_data('data.csv')
    
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
    plt.savefig('predictions_plot.png', dpi=100)
    print("\nPlot saved as 'predictions_plot.png'")
    plt.show()

if __name__ == "__main__":
    model, history = train_model(epochs=300, batch_size=8)
    predictions, y_test = evaluate_model(model)
    tflite_model = quantize_model(model)


