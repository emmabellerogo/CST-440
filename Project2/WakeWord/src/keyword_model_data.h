#ifndef KEYWORD_MODEL_DATA_H
#define KEYWORD_MODEL_DATA_H

/**
 * Keyword Spotting Model Data
 *
 * Model: CNN trained on Google Speech Commands v2
 * Input: 49 frames x 13 MFCCs (int8 quantized)
 * Output: 8 classes (int8 quantized)
 *
 * Classes:
 *   0: go
 *   1: stop
 *   2: up
 *   3: down
 *   4: yes
 *   5: no
 *   6: _silence_
 *   7: _unknown_
 */

extern const unsigned char keyword_model_data[];
extern const unsigned int keyword_model_data_len;

// Model configuration
#define MODEL_INPUT_FRAMES 49
#define MODEL_INPUT_MFCC 13
#define NUM_CLASSES 8

#endif // KEYWORD_MODEL_DATA_H
