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

// Data collection
const unsigned long COLLECTION_TIME = 10000; // 10 seconds in milliseconds
unsigned long start_time = 0;
bool collecting = true;
bool header_printed = false;

void setup() { 
  Serial.begin(115200); 
  while (!Serial); 
  
  // Load model 
  const tflite::Model* model = tflite::GetModel(trig_model_tflite); 
  if (model->version() != TFLITE_SCHEMA_VERSION) { 
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
    while (1); 
  } 
  
  input = interpreter->input(0); 
  output = interpreter->output(0); 
  
  // Print CSV header
  Serial.println("x,sin,cos,tan");
  header_printed = true;
  
  start_time = millis();
} 

void loop() { 
  if (collecting) {
    // Check if 10 seconds have passed
    if (millis() - start_time >= COLLECTION_TIME) {
      collecting = false;
      while(1); // Stop execution
    }
    
    static float x = 0.0f;
    
    // Input: x in radians 
    input->data.f[0] = x; 
    
    if (interpreter->Invoke() != kTfLiteOk) { 
      delay(1000); 
      return; 
    } 
    
    float sin_x = output->data.f[0]; 
    float cos_x = output->data.f[1]; 
    float tan_x = sin_x / (cos_x + 1e-6f); 
    
    // Output CSV row
    Serial.print(x, 6);
    Serial.print(",");
    Serial.print(sin_x, 6);
    Serial.print(",");
    Serial.print(cos_x, 6);
    Serial.print(",");
    Serial.println(tan_x, 6);
    
    x += 0.1f; 
    if (x > 6.283f) x = 0.0f; 
    
    delay(500); 
  }
}