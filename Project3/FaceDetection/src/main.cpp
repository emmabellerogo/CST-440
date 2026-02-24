/*
 * Face Detection — Arduino Nano 33 BLE Sense + ArduCAM Mini 2MP Plus (OV2640)
 *
 * Camera wiring (OV2640 → Nano 33 BLE Sense):
 *   VCC   → 5V    (module has onboard AMS1117-3.3 LDO; 5V in → 3.3V to sensor)
 *   GND   → GND
 *   SDA   → A4    (3.3V I2C — do NOT use 5V logic here)
 *   SCL   → A5    (3.3V I2C — do NOT use 5V logic here)
 *   MOSI  → 11    (3.3V SPI — signal lines are always 3.3V regardless of VCC)
 *   MISO  → 12    (3.3V SPI)
 *   SCK   → 13    (3.3V SPI)
 *   CS    → 10    (configurable — see CAM_CS below)
 *
 * Image pipeline:
 *   OV2640 QQVGA 160×120 RGB565 → stream FIFO row-by-row
 *   → convert RGB565 → grayscale (BT.601 integer luma)
 *   → center-crop to 96×96 written directly into TFLite input tensor
 *   No intermediate frame buffer needed — saves ~38 KB of SRAM.
 *
 * ⚠ Memory note: Nano 33 BLE Sense has 256 KB SRAM.
 *   kTensorArenaSize is set to 150 KB via TENSOR_ARENA_KB in platformio.ini.
 *
 * Model input:  uint8 [1,96,96,1]  (grayscale, quantized uint8)
 * Model output: uint8 (quantized sigmoid → dequantised to probability)
 */

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>

// memorysaver.h (project include/) defines OV2640_MINI_2MP_PLUS before the
// ArduCAM library reads it — this selects the correct sensor driver.
#include <memorysaver.h>
#include <ArduCAM.h>
// ArduCAM.h defines a 3-argument swap(type,i,j) macro that clashes with
// std::swap(a,b) used throughout the C++ standard library headers pulled in
// by TensorFlowLite.  Undefine it here before those headers are included.
#ifdef swap
#undef swap
#endif

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_config.h"
#include "face_model_data.h"

// ============================================================
// CAMERA CONFIG
// ArduCAM OV2640 in QQVGA (160×120) RGB565 mode.
// Pixels are read one-by-one from the SPI FIFO; only the 96×96
// center crop is kept — no full-frame buffer required.
// ============================================================

#define CAM_CS      10             // SPI chip-select — change if rerouted

#define CAM_WIDTH   160
#define CAM_HEIGHT  120

// Center-crop offsets: 160×120 → 96×96
#define CROP_X      ((CAM_WIDTH  - IMAGE_WIDTH)  / 2)  // 32 px
#define CROP_Y      ((CAM_HEIGHT - IMAGE_HEIGHT) / 2)  // 12 px

static ArduCAM myCAM(OV2640, CAM_CS);

// ============================================================
// TENSORFLOW LITE
// ============================================================

static tflite::AllOpsResolver      resolver;
static const tflite::Model*        tflModel    = nullptr;
static tflite::MicroInterpreter*   interpreter = nullptr;
static TfLiteTensor*               inputTensor  = nullptr;
static TfLiteTensor*               outputTensor = nullptr;

// Arena size set via TENSOR_ARENA_KB build flag in platformio.ini.
// nano33ble → 150 KB (256 KB SRAM; sufficient for the 96×96 Conv model)
#ifndef TENSOR_ARENA_KB
#  define TENSOR_ARENA_KB 6   // safe default for constrained boards
#endif
constexpr int kTensorArenaSize = TENSOR_ARENA_KB * 1024;
static uint8_t tensorArena[kTensorArenaSize] __attribute__((aligned(16)));

// ============================================================
// TFLite init
// ============================================================

bool initTFLite() {
  Serial.println(F("Initializing TFLite..."));

  tflModel = tflite::GetModel(face_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("ERROR: Model schema version mismatch!"));
    return false;
  }

  static tflite::MicroInterpreter staticInterpreter(
      tflModel, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("ERROR: AllocateTensors failed — arena too small."));
    Serial.print  (F("       Arena: "));
    Serial.print  (TENSOR_ARENA_KB);
    Serial.println(F(" KB.  Increase TENSOR_ARENA_KB in platformio.ini."));
    return false;
  }

  inputTensor  = interpreter->input(0);
  outputTensor = interpreter->output(0);

  if (inputTensor->dims->size != 4 ||
      inputTensor->dims->data[1] != IMAGE_HEIGHT ||
      inputTensor->dims->data[2] != IMAGE_WIDTH  ||
      inputTensor->dims->data[3] != IMAGE_CHANNELS) {
    Serial.println(F("ERROR: Input tensor shape mismatch — expected [1,96,96,1]."));
    return false;
  }

  Serial.print(F("  Arena used: "));
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(F(" bytes"));
  return true;
}

// ============================================================
// Camera init
// ============================================================

bool initCamera() {
  Serial.println(F("Initializing ArduCAM OV2640..."));

  Wire.begin();
  SPI.begin();
  pinMode(CAM_CS, OUTPUT);
  digitalWrite(CAM_CS, HIGH);

  // Verify SPI link to ArduCAM CPLD
  uint8_t r1 = 0, r2 = 0;
  for (int attempt = 0; attempt < 3; attempt++) {
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    r1 = myCAM.read_reg(ARDUCHIP_TEST1);
    myCAM.write_reg(ARDUCHIP_TEST1, 0xAA);
    r2 = myCAM.read_reg(ARDUCHIP_TEST1);
    if (r1 == 0x55 && r2 == 0xAA) break;
    delay(200);
  }

  if (r1 != 0x55 || r2 != 0xAA) {
    Serial.println(F("ERROR: ArduCAM SPI interface not responding!"));
    return false;
  }

  // --- Verify I2C link to OV2640 sensor ---
  uint8_t vid = 0, pid = 0;
  myCAM.wrSensorReg8_8(0xff, 0x01);            // select DSP register bank
  myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
  myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW,  &pid);
  if (vid != 0x26 || (pid != 0x41 && pid != 0x42)) {
    Serial.print(F("ERROR: OV2640 not found (vid=0x"));
    Serial.print(vid, HEX);
    Serial.print(F(", pid=0x"));
    Serial.print(pid, HEX);
    Serial.println(F(")"));
    Serial.println(F("  Check I2C wiring: SDA -> A4,  SCL -> A5"));
    Serial.println(F("  If vid=0x00: SDA not connected or I2C address wrong."));
    return false;
  }

  // --- Configure: QQVGA (160×120) RGB565, no JPEG compression ---
  // BMP mode gives raw RGB565 pixels in the FIFO so we can stream them
  // directly and convert to grayscale without a JPEG decoder.
  myCAM.set_format(BMP);
  myCAM.InitCAM();
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);
  delay(1000);  // allow AEC/AGC to settle

  Serial.println(F("  OV2640 ready (160x120 RGB565 BMP mode)."));
  return true;
}

// ============================================================
// Capture + stream directly into TFLite input tensor
//
// Reads the FIFO row-by-row (2 bytes per RGB565 pixel).
// Rows/columns outside the 96×96 centre crop are discarded.
// RGB565 → grayscale uses integer BT.601 luma coefficients:
//   Y = (77·R + 150·G + 29·B) >> 8   (coefficients sum to 256)
// No large intermediate buffer — ~400 bytes of stack at most.
// ============================================================

bool captureFrame() {
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  myCAM.start_capture();

  // Wait for capture-done flag (2 s timeout)
  const unsigned long deadline = millis() + 2000UL;
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() > deadline) {
      Serial.println(F("ERROR: Camera capture timed out!"));
      return false;
    }
  }

  uint8_t* dst = inputTensor->data.uint8;

  myCAM.CS_LOW();
  myCAM.set_fifo_burst();  // positions SPI for sequential FIFO reads

  for (int row = 0; row < CAM_HEIGHT; row++) {
    const bool rowInCrop = (row >= CROP_Y) && (row < CROP_Y + IMAGE_HEIGHT);

    for (int col = 0; col < CAM_WIDTH; col++) {
      // Read one RGB565 pixel (2 bytes, big-endian from ArduCAM FIFO)
      const uint8_t hi = SPI.transfer(0x00);
      const uint8_t lo = SPI.transfer(0x00);

      if (rowInCrop && (col >= CROP_X) && (col < CROP_X + IMAGE_WIDTH)) {
        // Unpack RGB565 channels to 8-bit
        const uint16_t px = ((uint16_t)hi << 8) | lo;
        const uint8_t  r  = (px >> 11) << 3;           // 5 → 8 bits
        const uint8_t  g  = ((px >> 5) & 0x3F) << 2;   // 6 → 8 bits
        const uint8_t  b  = (px & 0x1F) << 3;           // 5 → 8 bits

        // BT.601 integer luma (coefficients × 256 ≈ 77, 150, 29)
        const uint16_t luma = (uint16_t)77 * r + 150u * g + 29u * b;
        *dst++ = (uint8_t)(luma >> 8);
      }
    }
  }

  myCAM.CS_HIGH();
  return true;
}

// ============================================================
// Inference — returns 1=face, 0=no face, -1=error
// (inputTensor must already be populated by captureFrame)
// ============================================================

int runInference() {
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("ERROR: Invoke failed!"));
    return -1;
  }

  const float   scale      = outputTensor->params.scale;
  const int32_t zero_point = outputTensor->params.zero_point;
  const int32_t raw        = outputTensor->data.uint8[0];
  const float   prob       = (raw - zero_point) * scale;

  return (prob >= 0.5f) ? 1 : 0;
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  // Blink the built-in LED so we know the board booted even if Serial fails
  pinMode(LED_BUILTIN, OUTPUT);
  for (int i = 0; i < 6; i++) { digitalWrite(LED_BUILTIN, i % 2); delay(150); }

  Serial.begin(115200);
  delay(2000);  // give the USB CDC host time to attach

  Serial.println(F("\n================================"));
  Serial.println(F(" Face Detection — OV2640 + Nano33BLE"));
  Serial.println(F("================================"));

  if (!initCamera()) {
    while (1) { digitalWrite(LED_BUILTIN, HIGH); delay(100); digitalWrite(LED_BUILTIN, LOW); delay(100); }
  }

  if (!initTFLite()) {
    while (1) { digitalWrite(LED_BUILTIN, HIGH); delay(300); digitalWrite(LED_BUILTIN, LOW); delay(300); }
  }

  Serial.println(F("Ready — running inference every second."));
}

// ============================================================
// LOOP
// ============================================================

void loop() {
  // 1. Capture frame
  if (!captureFrame()) {
    delay(1000);
    return;
  }

  // 2. Inference
  const unsigned long t0     = millis();
  const int           result = runInference();
  const unsigned long dt     = millis() - t0;

  // 3. Print result
  if (result == 1) {
    Serial.print(F(">>> FACE DETECTED  prob="));
    Serial.print((outputTensor->data.uint8[0] - outputTensor->params.zero_point)
                  * outputTensor->params.scale, 3);
    Serial.print(F("  ("));
    Serial.print(dt);
    Serial.println(F(" ms)"));
  }

  delay(1000);
}
