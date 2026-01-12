import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load training data from CSV
def load_trig_data(csv_path='data.csv'):
    """Load sin, cos, tan data from CSV file"""
    df = pd.read_csv(csv_path)
    
    x_train = df['x'].values.astype(np.float32).reshape(-1, 1)
    y_train = df[['sin(x)', 'cos(x)', 'tan(x)']].values.astype(np.float32)
    
    print(f"Loaded {len(x_train)} samples from {csv_path}")
    return x_train, y_train

# Create a small model suitable for Arduino Nano
def create_trig_model():
    """Create a lightweight neural network for trig functions"""
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='tanh')  # 3 outputs: sin, cos, tan
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(epochs=100, batch_size=32):
    """Train the model on trig functions"""
    print("Loading training data...")
    x_train, y_train = load_trig_data('data.csv')
    
    print("Creating model...")
    model = create_trig_model()
    
    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    model.save('trig_model.keras')
    print("Model saved as 'trig_model.keras'")
    
    return model, history

def evaluate_model(model):
    """Evaluate model performance on test data"""
    x_test, y_test = load_trig_data('data.csv')
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Calculate errors for each function
    sin_error = np.mean(np.abs(predictions[:, 0] - y_test[:, 0]))
    cos_error = np.mean(np.abs(predictions[:, 1] - y_test[:, 1]))
    tan_error = np.mean(np.abs(predictions[:, 2] - y_test[:, 2]))
    
    print("\n=== Model Evaluation ===")
    print(f"Mean Absolute Error - Sin: {sin_error:.6f}")
    print(f"Mean Absolute Error - Cos: {cos_error:.6f}")
    print(f"Mean Absolute Error - Tan: {tan_error:.6f}")
    
    return predictions, y_test

if __name__ == "__main__":
    model, history = train_model(epochs=150, batch_size=32)
    predictions, y_test = evaluate_model(model)
    
    print("\nTraining complete! Files ready for deployment to Arduino.")
