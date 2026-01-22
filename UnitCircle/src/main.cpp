#include <Arduino.h> 
#include "trig_model1.h" 
#include <stdint.h> 

// Arduino TensorFlow Lite 
#include <TensorFlowLite.h> 
#include "tensorflow/lite/micro/micro_interpreter.h" 
#include "tensorflow/lite/micro/all_ops_resolver.h" 
#include "tensorflow/lite/schema/schema_generated.h" 

// Tensor arena (RAM for inference) 
constexpr int kTensorArenaSize = 8 * 1024; 
uint8_t tensor_arena[kTensorArenaSize]; 

tflite::MicroInterpreter* interpreter; 
TfLiteTensor* input; 
TfLiteTensor* output; 

void setup() { 
  Serial.begin(115200); 
  while (!Serial); 
  
  Serial.println("=== TensorFlow Lite Micro Trig Model ==="); 
  
  // Load model 
  const tflite::Model* model = tflite::GetModel(trig_model_tflite); 
  if (model->version() != TFLITE_SCHEMA_VERSION) { 
    Serial.println("ERROR: Model schema mismatch"); 
    while (1); 
  } 
  
  static tflite::AllOpsResolver resolver; 
  
  static tflite::MicroInterpreter static_interpreter( 
    model, 
    resolver, 
    tensor_arena, 
    kTensorArenaSize 
  ); 
  
  interpreter = &static_interpreter; 
  
  if (interpreter->AllocateTensors() != kTfLiteOk) { 
    Serial.println("ERROR: Tensor allocation failed"); 
    while (1); 
  } 
  
  input = interpreter->input(0); 
  output = interpreter->output(0); 
  
  Serial.println("Model loaded successfully!"); 
} 

void loop() { 
  static float x = 0.0f; 
  // Input: x in radians 
  input->data.f[0] = x; 
  
  if (interpreter->Invoke() != kTfLiteOk) { 
    Serial.println("ERROR: Inference failed"); 
    delay(1000); 
    return; } 
  
  float sin_x = output->data.f[0]; 
  float cos_x = output->data.f[1]; 
  float tan_x = sin_x / (cos_x + 1e-6f); 
  
  Serial.print("x="); 
  Serial.print(x, 3); 
  Serial.print(" | sin="); 
  Serial.print(sin_x, 5); 
  Serial.print(" | cos="); 
  Serial.print(cos_x, 5); 
  Serial.print(" | tan="); 
  Serial.println(tan_x, 5); 
  
  x += 0.1f; 
  if (x > 6.283f) x = 0.0f; 
  
  delay(500); 
}