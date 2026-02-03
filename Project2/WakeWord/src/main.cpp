/*
 * Keyword Detection for Arduino Nano 33 BLE Sense
 * Memory-optimized version
 *
 * Detects spoken keywords: go, stop, yes, no, up, down
 * Uses MFCC features + CNN model via TensorFlow Lite Micro
 */

#include <Arduino.h>
#include <PDM.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_config.h"
#include "keyword_model_data.h"

// ============================================================
// AUDIO CAPTURE
// ============================================================

// Audio buffer - collect 1 second of audio
static int16_t audioBuffer[AUDIO_LENGTH_SAMPLES];
static volatile int samplesCollected = 0;
static volatile bool audioReady = false;

// PDM buffer for callback
static const int PDM_BUFFER_SIZE = 256;

void onPDMdata() {
    int bytesAvailable = PDM.available();

    if (!audioReady && samplesCollected < AUDIO_LENGTH_SAMPLES) {
        int16_t tempBuffer[PDM_BUFFER_SIZE];
        int bytesToRead = min(bytesAvailable, (int)sizeof(tempBuffer));
        PDM.read(tempBuffer, bytesToRead);

        int samplesToRead = bytesToRead / 2;
        int spaceLeft = AUDIO_LENGTH_SAMPLES - samplesCollected;
        int samplesToCopy = min(samplesToRead, spaceLeft);

        memcpy(&audioBuffer[samplesCollected], tempBuffer, samplesToCopy * sizeof(int16_t));
        samplesCollected += samplesToCopy;

        if (samplesCollected >= AUDIO_LENGTH_SAMPLES) {
            audioReady = true;
        }
    } else {
        // Drain buffer to prevent overflow
        int16_t tempBuffer[PDM_BUFFER_SIZE];
        PDM.read(tempBuffer, bytesAvailable);
    }
}

// ============================================================
// DSP FUNCTIONS FOR MFCC (Memory Optimized)
// ============================================================

// Hanning window - computed once
static float hanningWindow[FRAME_LENGTH];

// Mel filter bank boundaries (computed once, stores only the bin indices)
static int melBinStart[NUM_MEL_BINS];
static int melBinCenter[NUM_MEL_BINS];
static int melBinEnd[NUM_MEL_BINS];

// DCT matrix
static float dctMatrix[NUM_MFCC][NUM_MEL_BINS];

static bool dspInitialized = false;

float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void initDSP() {
    if (dspInitialized) return;

    Serial.println("Initializing DSP...");

    // Initialize Hanning window
    for (int i = 0; i < FRAME_LENGTH; i++) {
        hanningWindow[i] = 0.5f - 0.5f * cosf(2.0f * PI * i / (FRAME_LENGTH - 1));
    }

    // Initialize mel filterbank bin indices only (not full matrix)
    float melLow = hzToMel(LOWER_FREQ);
    float melHigh = hzToMel(UPPER_FREQ);
    float melStep = (melHigh - melLow) / (NUM_MEL_BINS + 1);

    for (int i = 0; i < NUM_MEL_BINS; i++) {
        float melStart = melLow + i * melStep;
        float melCenter = melLow + (i + 1) * melStep;
        float melEnd = melLow + (i + 2) * melStep;

        melBinStart[i] = (int)floorf((FFT_LENGTH + 1) * melToHz(melStart) / SAMPLE_RATE);
        melBinCenter[i] = (int)floorf((FFT_LENGTH + 1) * melToHz(melCenter) / SAMPLE_RATE);
        melBinEnd[i] = (int)floorf((FFT_LENGTH + 1) * melToHz(melEnd) / SAMPLE_RATE);

        // Clamp to valid range
        if (melBinEnd[i] > FFT_LENGTH / 2) melBinEnd[i] = FFT_LENGTH / 2;
    }

    // Initialize DCT matrix (Type-II)
    for (int i = 0; i < NUM_MFCC; i++) {
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            dctMatrix[i][j] = cosf(PI * i * (j + 0.5f) / NUM_MEL_BINS);
        }
    }

    dspInitialized = true;
    Serial.println("DSP initialized.");
}

// Simple in-place FFT (Cooley-Tukey radix-2 DIT)
void fft(float* real, float* imag, int n) {
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
    for (int step = 2; step <= n; step *= 2) {
        float angle = -2.0f * PI / step;
        float wReal = 1.0f;
        float wImag = 0.0f;
        float wmReal = cosf(angle);
        float wmImag = sinf(angle);

        for (int m = 0; m < step / 2; m++) {
            for (int i = m; i < n; i += step) {
                int idx = i + step / 2;
                float tReal = wReal * real[idx] - wImag * imag[idx];
                float tImag = wReal * imag[idx] + wImag * real[idx];
                real[idx] = real[i] - tReal;
                imag[idx] = imag[i] - tImag;
                real[i] += tReal;
                imag[i] += tImag;
            }
            float newWReal = wReal * wmReal - wImag * wmImag;
            float newWImag = wReal * wmImag + wImag * wmReal;
            wReal = newWReal;
            wImag = newWImag;
        }
    }
}

// Compute mel filter weight on-the-fly (saves memory)
float getMelWeight(int melBin, int fftBin) {
    if (fftBin < melBinStart[melBin] || fftBin >= melBinEnd[melBin]) {
        return 0.0f;
    }

    if (fftBin < melBinCenter[melBin]) {
        // Rising edge
        return (float)(fftBin - melBinStart[melBin]) / (melBinCenter[melBin] - melBinStart[melBin]);
    } else {
        // Falling edge
        return (float)(melBinEnd[melBin] - fftBin) / (melBinEnd[melBin] - melBinCenter[melBin]);
    }
}

void extractMFCCFrame(float* frameAudio, float* mfccOut) {
    // FFT buffers (reused for each frame)
    static float fftReal[FFT_LENGTH];
    static float fftImag[FFT_LENGTH];

    // Apply window and prepare FFT input
    for (int i = 0; i < FFT_LENGTH; i++) {
        if (i < FRAME_LENGTH) {
            fftReal[i] = frameAudio[i] * hanningWindow[i];
        } else {
            fftReal[i] = 0.0f;
        }
        fftImag[i] = 0.0f;
    }

    // Compute FFT
    fft(fftReal, fftImag, FFT_LENGTH);

    // Apply mel filterbank and compute log mel energies
    static float melEnergies[NUM_MEL_BINS];

    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float energy = 0.0f;

        // Only iterate over non-zero filter range
        for (int k = melBinStart[m]; k < melBinEnd[m] && k <= FFT_LENGTH / 2; k++) {
            float magnitude = sqrtf(fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k]);
            float weight = getMelWeight(m, k);
            energy += magnitude * weight;
        }

        melEnergies[m] = logf(energy + 1e-6f);
    }

    // Apply DCT to get MFCCs
    for (int i = 0; i < NUM_MFCC; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            sum += melEnergies[j] * dctMatrix[i][j];
        }
        mfccOut[i] = sum;
    }
}

void extractMFCC(float* audio, float* mfccOutput) {
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        extractMFCCFrame(&audio[frame * FRAME_STEP], &mfccOutput[frame * NUM_MFCC]);
    }
}

// ============================================================
// TENSORFLOW LITE
// ============================================================

// TFLite globals
static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* inputTensor = nullptr;
static TfLiteTensor* outputTensor = nullptr;

// Reduced arena size
constexpr int kTensorArenaSize = 48 * 1024;
static uint8_t tensorArena[kTensorArenaSize] __attribute__((aligned(16)));

bool initTFLite() {
    Serial.println("Initializing TensorFlow Lite...");

    model = tflite::GetModel(keyword_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch!");
        return false;
    }

    static tflite::MicroInterpreter staticInterpreter(
        model, resolver, tensorArena, kTensorArenaSize);
    interpreter = &staticInterpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors failed!");
        return false;
    }

    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);

    Serial.print("  Arena used: ");
    Serial.print(interpreter->arena_used_bytes());
    Serial.println(" bytes");

    Serial.println("TensorFlow Lite initialized.");
    return true;
}

int runInference(float* mfccFeatures) {
    // Normalize features using training statistics
    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        for (int coef = 0; coef < NUM_MFCC; coef++) {
            int idx = frame * NUM_MFCC + coef;
            mfccFeatures[idx] = (mfccFeatures[idx] - NORM_MEAN[coef]) / NORM_STD[coef];
        }
    }

    // Copy to input tensor
    memcpy(inputTensor->data.f, mfccFeatures, NUM_FRAMES * NUM_MFCC * sizeof(float));

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("ERROR: Inference failed!");
        return -1;
    }

    // Find class with highest probability
    float maxScore = -1.0f;
    int maxIdx = 0;

    Serial.print("Scores: ");
    for (int i = 0; i < NUM_CLASSES; i++) {
        float score = outputTensor->data.f[i];
        Serial.print(CLASS_NAMES[i]);
        Serial.print("=");
        Serial.print(score, 2);
        Serial.print(" ");

        if (score > maxScore) {
            maxScore = score;
            maxIdx = i;
        }
    }
    Serial.println();

    return maxIdx;
}

// ============================================================
// MAIN
// ============================================================

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        delay(10);
    }

    Serial.println("\n========================================");
    Serial.println("Keyword Detection - Arduino Nano 33 BLE");
    Serial.println("========================================");
    Serial.println("Keywords: go, stop, yes, no, up, down");
    Serial.println();

    // Initialize DSP
    initDSP();

    // Initialize TensorFlow Lite
    if (!initTFLite()) {
        Serial.println("FATAL: TFLite initialization failed!");
        while (1) delay(1000);
    }

    // Initialize PDM microphone
    Serial.println("\nInitializing microphone...");
    PDM.onReceive(onPDMdata);

    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("FATAL: PDM initialization failed!");
        while (1) delay(1000);
    }

    Serial.println("Microphone initialized.");
    Serial.println("\n========================================");
    Serial.println("Ready! Speak a keyword...");
    Serial.println("========================================\n");
}

// Reuse buffer to save memory
static float audioFloat[AUDIO_LENGTH_SAMPLES];
static float mfccFeatures[NUM_FRAMES * NUM_MFCC];

void loop() {
    if (audioReady) {
        unsigned long startTime = millis();

        // Convert to float [-1, 1]
        for (int i = 0; i < AUDIO_LENGTH_SAMPLES; i++) {
            audioFloat[i] = audioBuffer[i] / 32768.0f;
        }

        // Check audio level
        float maxLevel = 0.0f;
        for (int i = 0; i < AUDIO_LENGTH_SAMPLES; i++) {
            float absVal = fabsf(audioFloat[i]);
            if (absVal > maxLevel) maxLevel = absVal;
        }

        if (maxLevel < 0.02f) {
            Serial.println("[silence - too quiet]");
        } else {
            // Extract MFCC features
            extractMFCC(audioFloat, mfccFeatures);

            // Run inference
            int predicted = runInference(mfccFeatures);

            unsigned long endTime = millis();

            Serial.print(">>> DETECTED: ");
            Serial.print(CLASS_NAMES[predicted]);
            Serial.print(" (");
            Serial.print(endTime - startTime);
            Serial.println(" ms)");
            Serial.println();
        }

        // Reset for next capture
        samplesCollected = 0;
        audioReady = false;
    }

    delay(10);
}
