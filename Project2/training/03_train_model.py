"""
Step 3: Train CNN Model for Keyword Spotting
Builds and trains a small CNN suitable for TinyML deployment.
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Training parameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ============================================================
# LOAD DATA
# ============================================================

def load_processed_data():
    """Load preprocessed dataset."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Load class names
    with open(os.path.join(PROCESSED_DIR, "classes.txt"), 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def create_tiny_cnn(input_shape, num_classes):
    """
    Create a small CNN model for keyword spotting.
    Designed to be small enough for microcontroller deployment.

    Architecture:
    - 2 Conv2D layers with small filters
    - MaxPooling for dimensionality reduction
    - Dense layers for classification
    - Optimized for ~20-50KB model size (quantized)
    """

    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Reshape for Conv2D (add channel dimension)
        layers.Reshape((*input_shape, 1)),

        # First Conv Block
        layers.Conv2D(8, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Second Conv Block
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Third Conv Block (optional - comment out if too large)
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def create_depthwise_cnn(input_shape, num_classes):
    """
    Alternative: Depthwise Separable CNN (more efficient).
    Use this if the regular CNN is too large.
    """

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),

        # Depthwise Separable Conv Block 1
        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(8, (1, 1), padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Depthwise Separable Conv Block 2
        layers.DepthwiseConv2D((3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(16, (1, 1), padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Dense layers
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_data, val_data):
    """Train the model with callbacks."""

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return history

def evaluate_model(model, test_data, classes):
    """Evaluate model on test set."""
    X_test, y_test = test_data

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Per-class accuracy
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(pred_classes[mask] == i)
            print(f"  {cls}: {class_acc:.4f} ({np.sum(mask)} samples)")

    return accuracy

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """Save model in multiple formats."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save Keras model
    keras_path = os.path.join(MODEL_DIR, "keyword_model.keras")
    model.save(keras_path)
    print(f"\nKeras model saved to {keras_path}")

    # Get model size
    model_size = os.path.getsize(keras_path)
    print(f"Keras model size: {model_size / 1024:.1f} KB")

    return keras_path

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Training CNN for Keyword Spotting")
    print("=" * 60)

    # Check if processed data exists
    if not os.path.exists(PROCESSED_DIR):
        print(f"\nError: Processed data not found at {PROCESSED_DIR}")
        print("Please run 02_preprocess_data.py first!")
        exit(1)

    # Load data
    print("\nLoading preprocessed data...")
    train_data, val_data, test_data, classes = load_processed_data()
    X_train, y_train = train_data

    print(f"Training samples: {len(X_train)}")
    print(f"Input shape: {X_train[0].shape}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")

    # Create model
    input_shape = X_train[0].shape
    num_classes = len(classes)

    print("\nCreating model...")
    model = create_tiny_cnn(input_shape, num_classes)
    model.summary()

    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated size (float32): {total_params * 4 / 1024:.1f} KB")
    print(f"Estimated size (int8): {total_params / 1024:.1f} KB")

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = train_model(model, train_data, val_data)

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    accuracy = evaluate_model(model, test_data, classes)

    # Save
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    save_model(model)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Next step: Run 04_convert_model.py")
    print("=" * 60)
