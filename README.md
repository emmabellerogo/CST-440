# CST-440: Embedded ML on Arduino

Team coursework (Josh Nelson, Emma Rogoveanu, Andru T., Martin B.) on running small neural networks on an Arduino Nano 33 BLE Sense with TensorFlow Lite Micro.

## Project 4: Hand gesture recognition
- Classifies three gestures (like, dislike, rock) from a 64x64 grayscale image, using HaGRID annotations. A MediaPipe hand-landmark gate runs before inference.
- Tiny CNN trained in TensorFlow/Keras, INT8-quantized with TFLite.
- **99.26% test accuracy** on 7,269 samples, in an **11.6 KB** model.
- Firmware in C++ (PlatformIO) in `Project4/gesture_project`. Full write-up: `Project4/Gesture_Detection_Report.ipynb`.

## Project 2: Speech keyword detection
- Detects six spoken keywords (go, stop, yes, no, up, down) on-device.
- Pipeline: PDM microphone capture, 13 MFCCs over 49 frames (1 second of 16 kHz audio), then a small CNN (depthwise-separable convolutions) running on TFLite Micro.
- Training scripts (preprocess, train, convert) are in `Project2/training`; firmware is in `Project2/WakeWord`. Background notes: `Project2/SpeecKeyword.md`.

## Project 1: Learning trig on a microcontroller
- A small network trained to predict sin and cos from an angle, quantized and run on the Arduino (`Project1/UnitCircle`), with `tan` derived as sin/cos.
- Accuracy tables and plots are in `Project1/*.csv` and `Project1/*.png`.

## Running the firmware
Each firmware folder is a PlatformIO project (`platformio.ini`). Open it in PlatformIO with an Arduino Nano 33 BLE Sense connected and upload.
