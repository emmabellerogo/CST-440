import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_model(model_path='trig_model.keras'):
    """Load the trained trig model"""
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Train the model first using train.py")
        return None

def predict_trig(angle, model=None):
    """
    Predict sin, cos, tan for a given angle
    
    Args:
        angle: Input angle in radians
        model: Trained model (loads if not provided)
    
    Returns:
        Dictionary with sin, cos, tan predictions
    """
    if model is None:
        model = load_model()
        if model is None:
            return None
    
    # Prepare input
    x = np.array([[angle]], dtype=np.float32)
    
    # Make prediction
    predictions = model.predict(x, verbose=0)[0]
    
    return {
        'angle': angle,
        'sin': float(predictions[0]),
        'cos': float(predictions[1]),
        'tan': float(predictions[2]),
        'actual_sin': float(np.sin(angle)),
        'actual_cos': float(np.cos(angle)),
        'actual_tan': float(np.tan(angle))
    }

def test_model_accuracy(model=None):
    """Test model accuracy on various angles"""
    if model is None:
        model = load_model()
        if model is None:
            return
    
    test_angles = np.linspace(-np.pi, np.pi, 20)
    
    print("\n=== Model Predictions vs Actual ===")
    print(f"{'Angle':<10} {'Sin Pred':<12} {'Sin Actual':<12} {'Error':<10}")
    print("-" * 44)
    
    total_error = 0
    for angle in test_angles:
        result = predict_trig(angle, model)
        error = abs(result['sin'] - result['actual_sin'])
        total_error += error
        print(f"{angle:<10.4f} {result['sin']:<12.6f} {result['actual_sin']:<12.6f} {error:<10.6f}")
    
    print(f"\nAverage Error: {total_error / len(test_angles):.6f}")

if __name__ == "__main__":
    model = load_model()
    
    if model is not None:
        # Test with some angles
        print("\n=== Testing Model ===")
        
        test_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
        
        for angle in test_angles:
            result = predict_trig(angle, model)
            print(f"\nAngle: {angle:.4f} rad ({np.degrees(angle):.1f}°)")
            print(f"  Sin - Predicted: {result['sin']:.6f}, Actual: {result['actual_sin']:.6f}")
            print(f"  Cos - Predicted: {result['cos']:.6f}, Actual: {result['actual_cos']:.6f}")
            print(f"  Tan - Predicted: {result['tan']:.6f}, Actual: {result['actual_tan']:.6f}")
        
        # Run full accuracy test
        test_model_accuracy(model)
