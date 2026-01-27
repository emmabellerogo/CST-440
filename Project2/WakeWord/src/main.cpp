#include <Arduino.h> 
#include "wake_word_model.h" 
#include <stdint.h> 

// Arduino TensorFlow Lite 
#include <TensorFlowLite.h> 
#include "tensorflow/lite/micro/micro_interpreter.h" 
#include "tensorflow/lite/micro/all_ops_resolver.h" 
#include "tensorflow/lite/schema/schema_generated.h" 

// For microphone input (PDM)
#include <PDM.h>

// Audio configuration
const int SAMPLE_RATE = 16000;        // 16 kHz sample rate
const int AUDIO_BUFFER_SIZE = 512;    // Buffer for audio samples
short audioBuffer[AUDIO_BUFFER_SIZE];
volatile int samplesRead = 0;

// Tensor arena (RAM for inference) 
constexpr int kTensorArenaSize = 20 * 1024; // Increased for audio model
uint8_t tensor_arena[kTensorArenaSize]; 

tflite::MicroInterpreter* interpreter; 
TfLiteTensor* input; 
TfLiteTensor* output; 

// Detection threshold
const float DETECTION_THRESHOLD = 0.7; // Adjust based on your model

void onPDMdata() {
  // Query the number of bytes available
  int bytesAvailable = PDM.available();
  
  // Read into the sample buffer
  PDM.read(audioBuffer, bytesAvailable);
  
  samplesRead = bytesAvailable / 2; // 2 bytes per sample
}

void setup() { 
  Serial.begin(115200); 
  while (!Serial); 
  
  Serial.println("Wake Word Detection Starting...");
  
  // Initialize PDM microphone
  PDM.onReceive(onPDMdata);
  PDM.setBufferSize(AUDIO_BUFFER_SIZE);
  
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
  
  // Load model 
  const tflite::Model* model = tflite::GetModel(wake_word_model_tflite); 
  if (model->version() != TFLITE_SCHEMA_VERSION) { 
    Serial.println("Model schema mismatch!");
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
    Serial.println("Failed to allocate tensors!");
    while (1); 
  } 
  
  input = interpreter->input(0); 
  output = interpreter->output(0); 
  
  // Print input shape information
  Serial.print("Input shape: ");
  Serial.print(input->dims->size);
  Serial.print(" dims = [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  Serial.println("Setup complete. Listening for wake word...");
} 

void preprocessAudio() {
  // Normalize audio samples to [-1, 1] range
  // This assumes your model expects normalized float inputs
  int inputSize = input->dims->data[1]; // Assuming shape [1, samples]
  
  for (int i = 0; i < inputSize && i < samplesRead; i++) {
    // Convert int16 to float and normalize
    input->data.f[i] = audioBuffer[i] / 32768.0f;
  }
  
  // Zero-pad if we don't have enough samples
  for (int i = samplesRead; i < inputSize; i++) {
    input->data.f[i] = 0.0f;
  }
}

void loop() { 
  // Wait for audio data
  if (samplesRead > 0) {
    
    // Preprocess audio data
    preprocessAudio();
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) { 
      Serial.println("Inference failed!");
      samplesRead = 0;
      return; 
    } 
    
    // Get prediction (assuming binary classification)
    float confidence = output->data.f[0];
    
    // Check if wake word detected
    if (confidence > DETECTION_THRESHOLD) {
      Serial.print("WAKE WORD DETECTED! Confidence: ");
      Serial.println(confidence, 4);
      
      // Visual feedback - built-in LED
      digitalWrite(LED_BUILTIN, HIGH);
      delay(500);
      digitalWrite(LED_BUILTIN, LOW);
    }
    
    // Optional: Print continuous confidence for debugging
    // Serial.print("Confidence: ");
    // Serial.println(confidence, 4);
    
    // Reset samples
    samplesRead = 0;
  }
  
  delay(10); // Small delay to prevent busy-waiting
}