// Configuration for 96x96 face detection (TFLite Micro)
// Must match training: train.py uses 96x96 grayscale, normalized [0,1], then quantized to uint8

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// Image dimensions - MUST MATCH TRAINING (train.py IMG_SIZE = 96)
#define IMAGE_WIDTH   96
#define IMAGE_HEIGHT  96
#define IMAGE_CHANNELS 1
#define IMAGE_SIZE    (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS)

// Model: binary classification (face vs no-face)
#define NUM_CLASSES   2

// Class names (index 0 = no face, 1 = face; output is single sigmoid quantized to uint8)
const char* const CLASS_NAMES[] = {
  "no_face",
  "face",
};

#endif // MODEL_CONFIG_H
