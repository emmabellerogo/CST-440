"""
Step 3: Train CNN Model for Keyword Spotting
Optimized for microcontroller deployment (small, fast, accurate)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "processed")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Early stopping patience
PATIENCE = 10

# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load preprocessed dataset."""
    print("Loading preprocessed data...")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Load class names
    with open(os.path.join(DATA_DIR, "classes.txt"), 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Classes: {classes}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def create_model(input_shape, num_classes):
    """
    Create a small CNN model optimized for microcontrollers.

    Architecture designed for:
    - Small memory footprint (~50KB quantized)
    - Fast inference
    - Good accuracy on keyword spotting

    Uses depthwise separable convolutions for efficiency.
    """

    model = keras.Sequential([
        # Input: (49, 13) MFCC features
        layers.Input(shape=input_shape),

        # Add channel dimension for Conv2D: (49, 13, 1)
        layers.Reshape((input_shape[0], input_shape[1], 1)),

        # First conv block - extract low-level features
        layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second conv block - depthwise separable for efficiency
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third conv block
        layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.25),

        # Global pooling - reduces to fixed size regardless of input
        layers.GlobalAveragePooling2D(),

        # Dense layers
        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.4),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_data, val_data):
    """Train the model with early stopping and learning rate reduction."""

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
        # Early stopping - stop if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print("\nTraining model...")
    print("="*50)

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    return history

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_data, classes):
    """Evaluate model on test set."""

    X_test, y_test = test_data

    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)

    # Overall accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Per-class accuracy
    predictions = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        mask = y_test == i
        if np.sum(mask) > 0:
            cls_acc = np.mean(pred_classes[mask] == i)
            print(f"  {cls}: {cls_acc*100:.1f}% ({np.sum(mask)} samples)")

    # Confusion matrix summary
    print("\nConfusion matrix (top confusions):")
    from collections import Counter
    confusions = Counter()
    for true, pred in zip(y_test, pred_classes):
        if true != pred:
            confusions[(classes[true], classes[pred])] += 1

    for (true_cls, pred_cls), count in confusions.most_common(5):
        print(f"  {true_cls} -> {pred_cls}: {count} times")

    return accuracy

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, classes, accuracy):
    """Save the trained model."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save Keras model
    keras_path = os.path.join(MODEL_DIR, "keyword_model.keras")
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")

    # Save model summary
    summary_path = os.path.join(MODEL_DIR, "model_summary.txt")
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f"\n\nTest Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Classes: {classes}\n")

    # Print model size info
    print("\nModel summary:")
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated size (float32): {total_params * 4 / 1024:.1f} KB")
    print(f"Estimated size (int8 quantized): {total_params / 1024:.1f} KB")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("STEP 3: Training CNN Model for Keyword Spotting")
    print("="*60)

    # Check data exists
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Processed data not found at {DATA_DIR}")
        print("Please run 01_preprocess_data.py first.")
        exit(1)

    # Load data
    train_data, val_data, test_data, classes = load_data()
    X_train, y_train = train_data

    # Get input shape and number of classes
    input_shape = X_train.shape[1:]  # (49, 13)
    num_classes = len(classes)

    print(f"\nInput shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Create model
    print("\nCreating model...")
    model = create_model(input_shape, num_classes)

    # Train
    history = train_model(model, train_data, val_data)

    # Evaluate
    accuracy = evaluate_model(model, test_data, classes)

    # Check if we hit target accuracy
    if accuracy >= 0.90:
        print("\n" + "="*50)
        print("SUCCESS: Achieved 90%+ accuracy!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print(f"Accuracy: {accuracy*100:.1f}% (target: 90%)")
        print("Consider: more data, longer training, or model tuning")
        print("="*50)

    # Save model
    save_model(model, classes, accuracy)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNext: Run 03_convert_model.py to convert for Arduino")
