What is speech keyword detection? 

Speech keyword detection (wake-word detection)  is a specialized form of speech recognition focussed on identifying short, predefined words or phrases (e.g. “hey device”, “start”, “stop”) rather than understanding full sentences. 

**Why keyword detection (instead of full ASR)?**

* Continuous listening at low power.   
* Minimal memory  and computer requirements.   
* Fast response time.   
* Privacy-preserving (no cloud dependency) 


 On microcontrollers, keyword spotting  speech-to-text. 

- The goal is classification, not transcription. 

**End-to-end wake word system architecture:** 

A speech keyword detection system on a microcontroller follows a streaming pipeline: 

* Audio Capture   
* Signal Processing   
* Feature Extraction  
* Neural Network Inference   
* Decision Logic/Output 

**Feature Extraction (Critical Step)** 

Most keyword models rely on: 

* MFCCs (Mel-Frequency Cepstral Coefficients)   
* Log-Mel Spectograms 

These compress time-domain audio into a compact frequency-domain representation aligned with human hearing. 

Feature extraction often runs on-device and consumes more CPU than inference itself. 

**Model Design for Keyword Detection:** 

Typical Input Shape: 

* (Time frames x Frequency bins)   
* Example: 49 x 10 MFCC matrix 

Output Layer: 

* Softmax classifier   
* One class per keyword \+ “unknown” \+ “Silence” 

**Why training happens off the microcontroller :** 

Microcontrollers cannot 

- Perform backpropagation   
- Store large datasets   
- Use GPUs 

**Training the model (off-device):**   
**Training Environment** 

- GPU-enabled workstation or cloud VM  
- Tensorflow/Keras   
- Large labeled speech dataset 

**Training Dataset Must Include:** 

- Target keywords (multiple speakers)   
- Background noise   
- Silence   
- Non-keyword speech

*Dataset diversity is more important than model depth for accuracy.* 

**Model Optimization for Deployment:**   
Before deployment, the trained model must be adapted for embedded constraints. 

**Optimization Techniques:** 

- Quantization   
- Pruning  
- Reduced input resolution   
- Smaller layer widths 

**Deployment on the Arduino:** 

- Audio driver   
- Feature Extractor   
- Inference engine   
- Output handler (LED, serial, display) 


LAB QUESTION 1  
Outline the phases of designing a wake-word recognizer, and the purpose of each.

Data Collection: Gather diverse audio samples of the wake word and non-wake words in different environments, accents, and noise levels.

Preprocessing: Clean audio (normalize volume, remove noise) and extract features such as spectrograms or MFCCs. 

Model Design: Select a lightweight neural network (CNN, RNN, or depthwise separable CNN) suitable for embedded systems. 

Training: Train on labeled data, validate accuracy, and fine-tune hyperparameters. 

Optimization: Apply quantization, pruning, or compression to reduce model size and memory usage. 

Deployment: Convert to a microcontroller-friendly format (e.g., TensorFlowlite), flash it onto the device, and test the performance. 

LAB QUESTION 2  
Provide an architectural diagram of an application for recognizing spoken commands. What is different on a microcontroller as opposed to a desktop or larger computing device?

Architecture:   
Input: Microphone captures audio   
Preprocessing: Convert audio to spectrogram/MFCC features   
Model Inference: Lightweight NN predicts command.   
Output Layer: Device action triggered (e.g., LED on/off, play sound). 

Microcontroller vs. Desktop Differences:   
Limited Memory/CPU.   
No GPU Acceleration.   
 

