#include "model_data.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// -------------------- Globals --------------------
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Tensor arena for TFLite Micro
constexpr int kTensorArenaSize = 4000;  // Increased size
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Total inferences per cycle and input range
constexpr int kInferencesPerCycle = 100;
constexpr float kXrange = 2.0f * 3.14159265f;  // 0 to 2π
}  // namespace

// -------------------- Setup --------------------
void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }  // Wait for serial connection
    
    Serial.println("Starting TensorFlow Lite Micro inference...");

    // Load model
    model = tflite::GetModel(trig_model_quantized_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("Model schema mismatch: ");
        Serial.println(model->version());
        return;
    }

    // Use MicroMutableOpResolver instead of AllOpsResolver (more memory efficient)
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddRelu();  // Add other ops your model uses
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Setup complete!");
    inference_count = 0;
}

// -------------------- Loop --------------------
void loop() {
    // Compute x in the training range
    float position = static_cast<float>(inference_count) / kInferencesPerCycle;
    float x = position * kXrange;

    // Quantize input
    int8_t x_quantized = static_cast<int8_t>(x / input->params.scale + input->params.zero_point);
    input->data.int8[0] = x_quantized;

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.print("Invoke failed for x=");
        Serial.println(x);
        return;
    }

    // Assuming output tensor has 2 values: [sin, cos]
    int8_t sin_quantized = output->data.int8[0];
    int8_t cos_quantized = output->data.int8[1];

    float sin_val = (sin_quantized - output->params.zero_point) * output->params.scale;
    float cos_val = (cos_quantized - output->params.zero_point) * output->params.scale;

    float tan_val = cos_val != 0.0f ? sin_val / cos_val : NAN;

    // Print results
    Serial.print("x = ");
    Serial.print(x, 4);
    Serial.print(" | sin_pred = ");
    Serial.print(sin_val, 4);
    Serial.print(" | cos_pred = ");
    Serial.print(cos_val, 4);
    Serial.print(" | tan_pred = ");
    Serial.println(tan_val, 4);

    inference_count = (inference_count + 1) % kInferencesPerCycle;
    delay(100);  // Small delay between inferences
}