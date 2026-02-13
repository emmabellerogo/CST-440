/*
 * Face Detection for Arduino (96x96 grayscale)
 * Uses TensorFlow Lite Micro with face_model.h from Project3/training
 *
 * Input: 96x96 grayscale image (uint8, 0-255). Model is INT8 quantized.
 * Output: Binary classification — face (1) or no_face (0).
 *
 * Camera: This project does not include a camera driver. The Nano 33 BLE Sense
 * has no camera. To get 96x96 images you can:
 *   - Use a board with a camera (e.g. ESP32-CAM) and add the appropriate library.
 *   - Send image data over serial from a PC or another device.
 *   - For testing, a placeholder image buffer is used below.
 */

#include <Arduino.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_config.h"
#include "face_model.h"

// ============================================================
// TENSORFLOW LITE
// ============================================================

static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* inputTensor = nullptr;
static TfLiteTensor* outputTensor = nullptr;

// Arena for TFLite (activations + intermediate tensors). Increase if AllocateTensors fails.
constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t tensorArena[kTensorArenaSize] __attribute__((aligned(16)));

// ============================================================
// IMAGE BUFFER (96x96 grayscale)
// ============================================================
// Replace this with your camera or serial input. For now we use a placeholder.
static uint8_t imageBuffer[IMAGE_SIZE];

// ============================================================
// TFLite init and inference
// ============================================================

bool initTFLite() {
  Serial.println("Initializing TensorFlow Lite (face model)...");

  // GetModel expects const void*; face_model_tflite may be non-const from xxd output
  model = tflite::GetModel(reinterpret_cast<const void*>(face_model_tflite));
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch!");
    return false;
  }

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed! Try increasing kTensorArenaSize.");
    return false;
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  // Sanity check input shape: expect [1, 96, 96, 1]
  if (inputTensor->dims->size != 4 ||
      inputTensor->dims->data[1] != IMAGE_HEIGHT ||
      inputTensor->dims->data[2] != IMAGE_WIDTH ||
      inputTensor->dims->data[3] != IMAGE_CHANNELS) {
    Serial.println("ERROR: Model input shape does not match 96x96x1");
    return false;
  }

  Serial.print("  Arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");
  Serial.println("TensorFlow Lite initialized.");
  return true;
}

// Returns 0 = no_face, 1 = face. Uses quantized uint8 input/output.
int runFaceInference(const uint8_t* image) {
  // Copy 96x96x1 into input tensor (quantized model expects uint8)
  memcpy(inputTensor->data.uint8, image, IMAGE_SIZE);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");
    return -1;
  }

  // Output is single uint8 (quantized sigmoid). Dequantize for thresholding.
  float scale = outputTensor->params.scale;
  int32_t zero_point = outputTensor->params.zero_point;
  int32_t raw = outputTensor->data.uint8[0];
  float prob = (raw - zero_point) * scale;

  // Threshold at 0.5 (match training binary classification)
  return (prob >= 0.5f) ? 1 : 0;
}

// ============================================================
// Placeholder: fill image buffer for testing (no camera)
// Replace this with your camera capture or serial-receive logic.
// ============================================================
void captureImagePlaceholder() {
  // Option A: zeros (no face) — model should predict no_face
  memset(imageBuffer, 0, IMAGE_SIZE);

  // Option B: simple pattern that might trigger "face" sometimes (for demo)
  // for (int i = 0; i < IMAGE_SIZE; i++) imageBuffer[i] = (i % 97) * 2;
}

// ============================================================
// SETUP & LOOP
// ============================================================

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println("\n========================================");
  Serial.println("Face Detection (96x96) - TFLite Micro");
  Serial.println("========================================");
  Serial.println("Image size: 96 x 96 grayscale");
  Serial.println("Model: face_model.h from Project3/training");
  Serial.println();

  if (!initTFLite()) {
    Serial.println("FATAL: TFLite init failed!");
    while (1) delay(1000);
  }

  Serial.println("\n========================================");
  Serial.println("Ready. Using placeholder image (no camera).");
  Serial.println("Replace captureImagePlaceholder() with your camera/serial input.");
  Serial.println("========================================\n");
}

void loop() {
  // Get 96x96 image (placeholder; replace with camera or serial)
  captureImagePlaceholder();

  unsigned long start = millis();
  int result = runFaceInference(imageBuffer);
  unsigned long elapsed = millis() - start;

  if (result >= 0) {
    Serial.print("Prediction: ");
    Serial.print(CLASS_NAMES[result]);
    Serial.print(" (");
    Serial.print(elapsed);
    Serial.println(" ms)");
  }

  delay(1000);
}
