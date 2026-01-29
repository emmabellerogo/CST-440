/**
 * Keyword Spotting on Arduino Nano 33 BLE Sense
 *
 * Recognizes 8 classes: go, stop, up, down, yes, no, silence, unknown
 * Uses MFCC features extracted from 1 second of audio
 */

#include <Arduino.h>
#include "keyword_model_data.h"
#include <stdint.h>
#include <math.h>

// Arduino TensorFlow Lite
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// For microphone input (PDM)
#include <PDM.h>

// ============================================================
// CONFIGURATION - Must match training parameters!
// ============================================================

// Audio parameters
const int SAMPLE_RATE = 16000;           // 16 kHz
const float AUDIO_LENGTH_SEC = 1.0f;     // 1 second of audio
const int AUDIO_LENGTH_SAMPLES = 16000;  // SAMPLE_RATE * AUDIO_LENGTH_SEC

// MFCC parameters (must match training!)
const int NUM_MFCC = 13;                 // Number of MFCC coefficients
const int FRAME_LENGTH = 640;            // 40ms at 16kHz
const int FRAME_STEP = 320;              // 20ms hop (50% overlap)
const int NUM_FRAMES = 49;               // Number of frames for 1 second
const int FFT_SIZE = 1024;               // FFT size (power of 2 >= FRAME_LENGTH)
const int NUM_MEL_BINS = 40;             // Mel filterbank bins

// Classification parameters
const float DETECTION_THRESHOLD = 0.6f;  // Minimum confidence to report

// Class labels (must match training order!)
const char* CLASS_LABELS[NUM_CLASSES] = {
    "go",        // 0
    "stop",      // 1
    "up",        // 2
    "down",      // 3
    "yes",       // 4
    "no",        // 5
    "_silence_", // 6
    "_unknown_"  // 7
};

// ============================================================
// BUFFERS
// ============================================================

// Audio buffer for 1 second of audio
int16_t audioBuffer[AUDIO_LENGTH_SAMPLES];
volatile int samplesCollected = 0;
volatile bool audioReady = false;

// Small buffer for PDM callback
const int PDM_BUFFER_SIZE = 512;
int16_t pdmBuffer[PDM_BUFFER_SIZE];

// MFCC output buffer
float mfccFeatures[NUM_FRAMES][NUM_MFCC];

// Working buffers for MFCC computation
float frame[FRAME_LENGTH];
float windowedFrame[FFT_SIZE];
float fftOutput[FFT_SIZE];
float melEnergies[NUM_MEL_BINS];
float dctOutput[NUM_MFCC];

// Mel filterbank (precomputed)
float melFilterbank[NUM_MEL_BINS][FFT_SIZE / 2 + 1];

// DCT matrix (precomputed)
float dctMatrix[NUM_MFCC][NUM_MEL_BINS];

// Hamming window (precomputed)
float hammingWindow[FRAME_LENGTH];

// ============================================================
// TENSORFLOW LITE
// ============================================================

// Tensor arena - adjust size based on model needs
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Quantization parameters
float input_scale = 1.0f;
int input_zero_point = 0;
float output_scale = 1.0f;
int output_zero_point = 0;

// ============================================================
// HELPER FUNCTIONS
// ============================================================

// Convert frequency to Mel scale
float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Convert Mel scale to frequency
float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Initialize Hamming window
void initHammingWindow() {
    for (int i = 0; i < FRAME_LENGTH; i++) {
        hammingWindow[i] = 0.54f - 0.46f * cosf(2.0f * PI * i / (FRAME_LENGTH - 1));
    }
}

// Initialize Mel filterbank
void initMelFilterbank() {
    float melMin = hzToMel(20.0f);
    float melMax = hzToMel(SAMPLE_RATE / 2.0f);

    // Mel points
    float melPoints[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        melPoints[i] = melMin + i * (melMax - melMin) / (NUM_MEL_BINS + 1);
    }

    // Convert to Hz and then to FFT bins
    int binPoints[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        float hz = melToHz(melPoints[i]);
        binPoints[i] = (int)floorf((FFT_SIZE + 1) * hz / SAMPLE_RATE);
    }

    // Create filterbank
    for (int i = 0; i < NUM_MEL_BINS; i++) {
        for (int j = 0; j < FFT_SIZE / 2 + 1; j++) {
            melFilterbank[i][j] = 0.0f;

            if (j >= binPoints[i] && j <= binPoints[i + 1]) {
                melFilterbank[i][j] = (float)(j - binPoints[i]) / (binPoints[i + 1] - binPoints[i]);
            } else if (j >= binPoints[i + 1] && j <= binPoints[i + 2]) {
                melFilterbank[i][j] = (float)(binPoints[i + 2] - j) / (binPoints[i + 2] - binPoints[i + 1]);
            }
        }
    }
}

// Initialize DCT matrix for MFCC
void initDCTMatrix() {
    for (int i = 0; i < NUM_MFCC; i++) {
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            dctMatrix[i][j] = cosf(PI * i * (j + 0.5f) / NUM_MEL_BINS);
        }
    }
}

// Simple FFT implementation (Cooley-Tukey radix-2)
// Note: For production, use ARM CMSIS-DSP for better performance
void computeFFT(float* real, float* imag, int n) {
    // Bit reversal
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tempR = real[i];
            float tempI = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tempR;
            imag[j] = tempI;
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // FFT computation
    for (int stage = 1; stage < n; stage *= 2) {
        float angle = -PI / stage;
        float wR = cosf(angle);
        float wI = sinf(angle);

        for (int i = 0; i < n; i += stage * 2) {
            float tR = 1.0f, tI = 0.0f;
            for (int k = 0; k < stage; k++) {
                int idx1 = i + k;
                int idx2 = i + k + stage;

                float uR = real[idx1];
                float uI = imag[idx1];
                float vR = real[idx2] * tR - imag[idx2] * tI;
                float vI = real[idx2] * tI + imag[idx2] * tR;

                real[idx1] = uR + vR;
                imag[idx1] = uI + vI;
                real[idx2] = uR - vR;
                imag[idx2] = uI - vI;

                float newTR = tR * wR - tI * wI;
                tI = tR * wI + tI * wR;
                tR = newTR;
            }
        }
    }
}

// Compute MFCC for a single frame
void computeFrameMFCC(int16_t* audioFrame, float* mfcc) {
    static float fftImag[FFT_SIZE];

    // Apply window and copy to FFT buffer
    for (int i = 0; i < FFT_SIZE; i++) {
        if (i < FRAME_LENGTH) {
            windowedFrame[i] = audioFrame[i] / 32768.0f * hammingWindow[i];
        } else {
            windowedFrame[i] = 0.0f;
        }
        fftImag[i] = 0.0f;
    }

    // Compute FFT
    computeFFT(windowedFrame, fftImag, FFT_SIZE);

    // Compute power spectrum
    for (int i = 0; i <= FFT_SIZE / 2; i++) {
        fftOutput[i] = windowedFrame[i] * windowedFrame[i] + fftImag[i] * fftImag[i];
    }

    // Apply Mel filterbank
    for (int i = 0; i < NUM_MEL_BINS; i++) {
        melEnergies[i] = 0.0f;
        for (int j = 0; j <= FFT_SIZE / 2; j++) {
            melEnergies[i] += fftOutput[j] * melFilterbank[i][j];
        }
        // Log compression
        melEnergies[i] = logf(melEnergies[i] + 1e-6f);
    }

    // Apply DCT to get MFCCs
    for (int i = 0; i < NUM_MFCC; i++) {
        mfcc[i] = 0.0f;
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            mfcc[i] += melEnergies[j] * dctMatrix[i][j];
        }
    }
}

// Compute all MFCCs for the audio buffer
void computeMFCC() {
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int startSample = frame * FRAME_STEP;
        computeFrameMFCC(&audioBuffer[startSample], mfccFeatures[frame]);
    }
}

// ============================================================
// PDM CALLBACK
// ============================================================

void onPDMdata() {
    int bytesAvailable = PDM.available();
    int samplesToRead = bytesAvailable / 2;

    if (audioReady) return; // Skip if previous audio not processed

    PDM.read(pdmBuffer, bytesAvailable);

    // Copy to main buffer
    for (int i = 0; i < samplesToRead && samplesCollected < AUDIO_LENGTH_SAMPLES; i++) {
        audioBuffer[samplesCollected++] = pdmBuffer[i];
    }

    // Check if we have 1 second of audio
    if (samplesCollected >= AUDIO_LENGTH_SAMPLES) {
        audioReady = true;
    }
}

// ============================================================
// SETUP
// ============================================================

void setup() {
    Serial.begin(115200);
    while (!Serial);

    Serial.println("========================================");
    Serial.println("Keyword Spotting - Arduino Nano 33 BLE");
    Serial.println("========================================");
    Serial.println();

    // Initialize LED
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // Initialize precomputed tables
    Serial.println("Initializing MFCC tables...");
    initHammingWindow();
    initMelFilterbank();
    initDCTMatrix();

    // Initialize PDM microphone
    Serial.println("Initializing microphone...");
    PDM.onReceive(onPDMdata);
    PDM.setBufferSize(PDM_BUFFER_SIZE * 2);

    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("ERROR: Failed to start PDM microphone!");
        while (1) {
            digitalWrite(LED_BUILTIN, HIGH);
            delay(100);
            digitalWrite(LED_BUILTIN, LOW);
            delay(100);
        }
    }

    // Load TensorFlow Lite model
    Serial.println("Loading model...");
    const tflite::Model* model = tflite::GetModel(keyword_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("ERROR: Model schema version mismatch! Expected ");
        Serial.print(TFLITE_SCHEMA_VERSION);
        Serial.print(", got ");
        Serial.println(model->version());
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
        Serial.println("ERROR: Failed to allocate tensors!");
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // Get quantization parameters
    if (input->type == kTfLiteInt8) {
        input_scale = input->params.scale;
        input_zero_point = input->params.zero_point;
    }
    if (output->type == kTfLiteInt8) {
        output_scale = output->params.scale;
        output_zero_point = output->params.zero_point;
    }

    // Print model info
    Serial.println();
    Serial.println("Model loaded successfully!");
    Serial.print("  Input shape: [");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        if (i < input->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");
    Serial.print("  Input type: ");
    Serial.println(input->type == kTfLiteInt8 ? "int8" : "float32");

    Serial.print("  Output shape: [");
    for (int i = 0; i < output->dims->size; i++) {
        Serial.print(output->dims->data[i]);
        if (i < output->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");

    Serial.println();
    Serial.println("Classes:");
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.print("  ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(CLASS_LABELS[i]);
    }

    Serial.println();
    Serial.println("========================================");
    Serial.println("Listening for keywords...");
    Serial.println("========================================");
    Serial.println();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
    if (audioReady) {
        // LED on during processing
        digitalWrite(LED_BUILTIN, HIGH);

        // Compute MFCC features
        computeMFCC();

        // Copy features to input tensor
        if (input->type == kTfLiteInt8) {
            // Quantize input
            int8_t* input_data = input->data.int8;
            for (int f = 0; f < NUM_FRAMES; f++) {
                for (int c = 0; c < NUM_MFCC; c++) {
                    float val = mfccFeatures[f][c];
                    int32_t quantized = (int32_t)roundf(val / input_scale) + input_zero_point;
                    quantized = max(-128, min(127, quantized));
                    input_data[f * NUM_MFCC + c] = (int8_t)quantized;
                }
            }
        } else {
            // Float input
            float* input_data = input->data.f;
            for (int f = 0; f < NUM_FRAMES; f++) {
                for (int c = 0; c < NUM_MFCC; c++) {
                    input_data[f * NUM_MFCC + c] = mfccFeatures[f][c];
                }
            }
        }

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("ERROR: Inference failed!");
        } else {
            // Get predictions
            float predictions[NUM_CLASSES];

            if (output->type == kTfLiteInt8) {
                // Dequantize output
                int8_t* output_data = output->data.int8;
                for (int i = 0; i < NUM_CLASSES; i++) {
                    predictions[i] = (output_data[i] - output_zero_point) * output_scale;
                }
            } else {
                // Float output
                for (int i = 0; i < NUM_CLASSES; i++) {
                    predictions[i] = output->data.f[i];
                }
            }

            // Find best prediction
            int bestClass = 0;
            float bestScore = predictions[0];
            for (int i = 1; i < NUM_CLASSES; i++) {
                if (predictions[i] > bestScore) {
                    bestScore = predictions[i];
                    bestClass = i;
                }
            }

            // Report if confidence above threshold and not silence/unknown
            if (bestScore > DETECTION_THRESHOLD && bestClass < 6) {
                Serial.print(">>> DETECTED: ");
                Serial.print(CLASS_LABELS[bestClass]);
                Serial.print(" (confidence: ");
                Serial.print(bestScore * 100, 1);
                Serial.println("%)");

                // Flash LED for detected keyword
                for (int i = 0; i < 3; i++) {
                    digitalWrite(LED_BUILTIN, LOW);
                    delay(50);
                    digitalWrite(LED_BUILTIN, HIGH);
                    delay(50);
                }
            }

            // Debug: print all scores periodically
            static int debugCounter = 0;
            if (++debugCounter >= 5) {
                debugCounter = 0;
                Serial.println("--- Scores ---");
                for (int i = 0; i < NUM_CLASSES; i++) {
                    Serial.print("  ");
                    Serial.print(CLASS_LABELS[i]);
                    Serial.print(": ");
                    Serial.print(predictions[i] * 100, 1);
                    Serial.println("%");
                }
                Serial.println();
            }
        }

        // Reset for next audio capture
        digitalWrite(LED_BUILTIN, LOW);
        samplesCollected = 0;
        audioReady = false;
    }

    delay(10);
}
