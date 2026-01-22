import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure your serial port
SERIAL_PORT = 'COM3'  # Change to your port (e.g., '/dev/ttyUSB0' on Linux, '/dev/cu.usbserial-*' on Mac)
BAUD_RATE = 115200

def capture_data_from_arduino(port, baudrate, timeout=15):
    """
    Captures data from Arduino and saves to CSV
    """
    print(f"Connecting to {port}...")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection to establish
        
        print("Connected! Waiting for data collection...")
        
        csv_data = []
        recording = False
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line:
                print(line)  # Echo to console
                
                if "=== CSV DATA START ===" in line:
                    recording = True
                    continue
                    
                if "=== CSV DATA END ===" in line:
                    recording = False
                    break
                    
                if recording:
                    csv_data.append(line)
        
        ser.close()
        
        # Save raw CSV
        with open('trig_data.csv', 'w') as f:
            f.write('\n'.join(csv_data))
        
        print("\nData saved to 'trig_data.csv'")
        return 'trig_data.csv'
        
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def analyze_data(csv_file):
    """
    Analyze the trigonometric data and create visualizations
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    print("\n=== Data Summary ===")
    print(df.describe())
    
    # Calculate true values for comparison
    df['sin_true'] = np.sin(df['x'])
    df['cos_true'] = np.cos(df['x'])
    df['tan_true'] = np.tan(df['x'])
    
    # Calculate errors
    df['sin_error'] = np.abs(df['sin'] - df['sin_true'])
    df['cos_error'] = np.abs(df['cos'] - df['cos_true'])
    df['tan_error'] = np.abs(df['tan'] - df['tan_true'])
    
    print("\n=== Mean Absolute Errors ===")
    print(f"Sin Error: {df['sin_error'].mean():.6f}")
    print(f"Cos Error: {df['cos_error'].mean():.6f}")
    print(f"Tan Error: {df['tan_error'].mean():.6f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sin predictions vs true
    axes[0, 0].plot(df['x'], df['sin'], 'o-', label='Model Prediction', alpha=0.7)
    axes[0, 0].plot(df['x'], df['sin_true'], '--', label='True Value', alpha=0.7)
    axes[0, 0].set_xlabel('x (radians)')
    axes[0, 0].set_ylabel('sin(x)')
    axes[0, 0].set_title('Sine Function: Model vs True')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cos predictions vs true
    axes[0, 1].plot(df['x'], df['cos'], 'o-', label='Model Prediction', alpha=0.7)
    axes[0, 1].plot(df['x'], df['cos_true'], '--', label='True Value', alpha=0.7)
    axes[0, 1].set_xlabel('x (radians)')
    axes[0, 1].set_ylabel('cos(x)')
    axes[0, 1].set_title('Cosine Function: Model vs True')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error over x
    axes[1, 0].plot(df['x'], df['sin_error'], 'o-', label='Sin Error', alpha=0.7)
    axes[1, 0].plot(df['x'], df['cos_error'], 'o-', label='Cos Error', alpha=0.7)
    axes[1, 0].set_xlabel('x (radians)')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Prediction Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Tan comparison (limited range to avoid infinities)
    tan_mask = np.abs(df['tan']) < 10  # Filter out large tan values
    axes[1, 1].plot(df.loc[tan_mask, 'x'], df.loc[tan_mask, 'tan'], 'o-', 
                    label='Model Prediction', alpha=0.7)
    axes[1, 1].plot(df.loc[tan_mask, 'x'], df.loc[tan_mask, 'tan_true'], '--', 
                    label='True Value', alpha=0.7)
    axes[1, 1].set_xlabel('x (radians)')
    axes[1, 1].set_ylabel('tan(x)')
    axes[1, 1].set_title('Tangent Function: Model vs True')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trig_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'trig_analysis.png'")
    plt.show()

if __name__ == "__main__":
    # Capture data from Arduino
    csv_file = capture_data_from_arduino(SERIAL_PORT, BAUD_RATE)
    
    if csv_file:
        # Analyze the data
        analyze_data(csv_file)
    else:
        print("Failed to capture data from Arduino")