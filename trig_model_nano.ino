/*
 * Trigonometric Function ML Model - Arduino Nano Deployment
 * 
 * This sketch demonstrates running a neural network on Arduino Nano
 * to approximate sin(x) and cos(x) functions.
 * 
 * Hardware: Arduino Nano (ATmega328P)
 * Model: TensorFlow Lite (quantized, 8-bit)
 * 
 * Serial Commands:
 *   - Send angle in radians (e.g., "1.57" for π/2)
 *   - Model outputs: sin(x) and cos(x)
 */

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "trig_model.h"  // Include generated model header

// TensorFlow Lite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena for model (adjust size if needed)
  constexpr int kTensorArenaSize = 4 * 1024;  // 4KB
  uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("====================================");
  Serial.println("Trig Function ML Model - Arduino Nano");
  Serial.println("====================================");

  // Load model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model version mismatch!");
    return;
  }
  Serial.println("Model loaded successfully");

  // Create resolver for ops
  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: Tensor allocation failed!");
    return;
  }
  Serial.println("Tensors allocated");

  // Get pointers to input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("
Ready! Send angle in radians via Serial Monitor");
  Serial.println("Example: 1.5708 (for π/2 = 90°)");
  Serial.println("====================================");
}

void loop() {
  if (Serial.available() > 0) {
    // Read angle from serial
    float angle = Serial.parseFloat();

    // Normalize input: x_norm = mod(x, 2π) / 2π
    float normalized = fmod(angle, 2.0 * PI) / (2.0 * PI);

    // Set input tensor
    input->data.f[0] = normalized;

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("ERROR: Inference failed!");
      return;
    }

    // Get outputs
    float sin_pred = output->data.f[0];
    float cos_pred = output->data.f[1];

    // Calculate actual values for comparison
    float sin_actual = sin(angle);
    float cos_actual = cos(angle);

    // Display results
    Serial.println("
--- Results ---");
    Serial.print("Input angle: ");
    Serial.print(angle, 4);
    Serial.print(" rad (");
    Serial.print(angle * 180.0 / PI, 2);
    Serial.println("°)");

    Serial.println("
Predicted:");
    Serial.print("  sin(x) = ");
    Serial.println(sin_pred, 6);
    Serial.print("  cos(x) = ");
    Serial.println(cos_pred, 6);

    Serial.println("
Actual:");
    Serial.print("  sin(x) = ");
    Serial.println(sin_actual, 6);
    Serial.print("  cos(x) = ");
    Serial.println(cos_actual, 6);

    Serial.println("
Error:");
    Serial.print("  sin: ");
    Serial.println(abs(sin_pred - sin_actual), 6);
    Serial.print("  cos: ");
    Serial.println(abs(cos_pred - cos_actual), 6);
    Serial.println("===============");
  }
}
