# Face Detection — Nano 33 BLE Sense + OV7670 Setup Guide

## Overview

This project runs a TFLite INT8 face detection model on an **Arduino Nano 33 BLE Sense**
using an **OV7670** camera module wired over a breadboard.

- The OV7670 captures a 160×120 grayscale frame (QQVGA)
- The Nano center-crops it to 96×96 to match the trained model's input shape
- TFLite Micro runs inference and prints `face` or `no_face` over Serial

---

## What You Need

| Item | Notes |
|---|---|
| Arduino Nano 33 BLE Sense | Not the Nano 33 IoT — must be the **BLE Sense** |
| OV7670 camera module | ~$3–5, available on Amazon/AliExpress |
| Breadboard | Full-size recommended |
| Jumper wires | Male-to-male, at least 18 |
| USB-A to Micro-USB cable | For flashing and Serial monitor |
| PC with PlatformIO installed | VS Code + PlatformIO extension |

---

## Step 1 — Install PlatformIO

1. Open **VS Code**
2. Go to the Extensions panel (`Ctrl+Shift+X`)
3. Search for **PlatformIO IDE** and install it
4. Restart VS Code when prompted

---

## Step 2 — Wire the OV7670 to the Nano

> ⚠️ **Use 3.3V only.** The OV7670 is NOT 5V tolerant. Connecting 5V will destroy it.

> ⚠️ **D0 and D1** on the Nano are shared with USB Serial (RX/TX). Disconnect the USB
> cable after wiring before flashing, then reconnect after upload.

Wire the following connections on the breadboard:

| OV7670 Pin | Nano 33 BLE Sense Pin | Purpose |
|---|---|---|
| 3.3V | 3.3V | Power |
| GND | GND | Ground |
| SDA | A4 | I2C — camera config |
| SCL | A5 | I2C — camera config |
| VSYNC | D8 | Frame sync |
| HREF | A1 | Row valid signal |
| PCLK | A0 | Pixel clock |
| XCLK | D9 | Master clock (driven by library) |
| D0 | D0 (RX) | Pixel data bit 0 |
| D1 | D1 (TX) | Pixel data bit 1 |
| D2 | D10 | Pixel data bit 2 |
| D3 | D12 | Pixel data bit 3 |
| D4 | D11 | Pixel data bit 4 |
| D5 | D13 | Pixel data bit 5 |
| D6 | A2 | Pixel data bit 6 |
| D7 | A3 | Pixel data bit 7 |
| RESET | 3.3V | Tie high (always on) |
| PWDN | GND | Tie low (always on) |

### Breadboard Diagram (text)

```
OV7670                    Nano 33 BLE Sense
───────                   ─────────────────
3.3V  ──────────────────► 3.3V
GND   ──────────────────► GND
SDA   ──────────────────► A4
SCL   ──────────────────► A5
VSYNC ──────────────────► D8
HREF  ──────────────────► A1
PCLK  ──────────────────► A0
XCLK  ──────────────────► D9
D0    ──────────────────► D0  ⚠️ (disconnect USB to flash)
D1    ──────────────────► D1  ⚠️ (disconnect USB to flash)
D2    ──────────────────► D10
D3    ──────────────────► D12
D4    ──────────────────► D11
D5    ──────────────────► D13
D6    ──────────────────► A2
D7    ──────────────────► A3
RESET ──────────────────► 3.3V (tie directly on breadboard)
PWDN  ──────────────────► GND  (tie directly on breadboard)
```

---

## Step 3 — Retrain the Model (if not done already)

From `Project3/training/`:

```bash
cd Project3/training
python3 train.py
```

This produces:
- `face_model.tflite` — INT8 quantized model for the Nano
- `face_model_float32.tflite` — float32 model for Python testing
- `face_model.h` — C header (not used directly; see Step 4)

---

## Step 4 — Regenerate the Model Header

The Arduino project uses `src/face_model_data.h`. Regenerate it any time you retrain:

```bash
cd Project3/training
xxd -i face_model.tflite \
  | sed 's/unsigned char face_model_tflite/alignas(8) const unsigned char face_model_data/g; \
         s/unsigned int face_model_tflite_len/const unsigned int face_model_data_len/g' \
  > ../FaceDetection/src/face_model_data.h
```

Then manually wrap it with the header guard — or just re-run the existing `testing.py`
pipeline which triggers `train.py` to do this automatically.

---

## Step 5 — Flash the Nano

1. Open the `Project3/FaceDetection/` folder in VS Code with PlatformIO
2. **Disconnect the USB cable** (because D0/D1 are wired to the camera)
3. Reconnect USB — PlatformIO will detect the Nano
4. Click the **Upload** button (→ arrow) in the PlatformIO toolbar, or run:

```bash
pio run --target upload
```

5. If upload fails with a port error, press and hold the Nano's reset button, then
   release it right as PlatformIO begins uploading (puts it into bootloader mode)

---

## Step 6 — Open the Serial Monitor

Once flashed, reconnect USB and open the Serial monitor at **115200 baud**:

```bash
pio device monitor --baud 115200
```

Or use the PlatformIO Serial Monitor button in the VS Code status bar.

Expected output every ~1 second:

```
================================
 Face Detection — OV7670 + TFLite
================================
Initializing OV7670...
  OV7670 ready.
Initializing TFLite...
  Arena used: XXXXX bytes
Ready — running inference every second.
  Probability: 0.923
Prediction: face  (284 ms)
  Probability: 0.031
Prediction: no_face  (281 ms)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Camera.begin() failed` | Bad wiring | Double-check SDA→A4, SCL→A5, VSYNC→D8 |
| `AllocateTensors failed` | Arena too small | Increase `kTensorArenaSize` in `main.cpp` |
| `Model schema version mismatch` | Old model header | Regenerate `face_model_data.h` (Step 4) |
| Always predicts `no_face` | INT8 quant issue | Retrain with more data; verify 96×96 crop |
| Upload fails | D0/D1 conflict | Disconnect camera D0/D1 wires, flash, rewire |
| Garbled Serial output | Wrong baud rate | Set monitor to **115200** |

---

## File Reference

```
Project3/
├── training/
│   ├── train.py              # Train the model
│   ├── testing.py            # Test with webcam (Python, uses float32 model)
│   ├── face_model.tflite     # INT8 model → goes to Nano
│   └── face_model_float32.tflite  # float32 model → Python testing only
└── FaceDetection/
    ├── platformio.ini        # Board config + library deps
    └── src/
        ├── main.cpp          # OV7670 capture + TFLite inference
        ├── model_config.h    # Image dimensions + class names
        └── face_model_data.h # Model weights as C byte array
```

---

## Changing the Camera Later

If you end up with a different camera module (e.g. OV2640, HM01B0), the changes needed are:

1. Update `lib_deps` in `platformio.ini` with the correct library
2. Replace `Camera.begin()` and `Camera.readFrame()` calls in `main.cpp`
3. Adjust `CAM_WIDTH` / `CAM_HEIGHT` and crop offsets if the resolution differs
4. Everything else (TFLite inference, model header) stays the same
