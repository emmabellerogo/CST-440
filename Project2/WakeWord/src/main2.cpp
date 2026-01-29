#include <PDM.h>

// Default number of output channels
static const char channels = 1;

// default PCM output frequency
static const int frequency = 16000;

// Buffer size for reading samples
const int BUFFER_SIZE = 256;

// Buffer to read samples into, each sample is 16-bit
int16_t sampleBuffer[BUFFER_SIZE];

// Number of audio samples read
volatile int samplesRead;

void setup() {
  Serial.begin(9600);
  while (!Serial); // Wait for serial connection

  // Configure the data receive callback
  PDM.onReceive(onPDMdata);
// optionally set the gain, defaults to 20
// PDM.setGain(30);

  // Initialize PDM with:
  // - one channel (mono mode)
  // - a 22050 Hz sample rate (adjust as needed)
  if (!PDM.begin(channels, frequency)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

  // Adjust the gain as needed
  PDM.setGain(30);
}

void loop() {
  // Wait for samples to be read
  if (samplesRead) {
    // Send the audio data over the serial port as comma-separated values
    for (int i = 0; i < samplesRead; i++) {
      Serial.print(sampleBuffer[i]);
      if (i < samplesRead - 1) {
        Serial.print(",");
      }
    }
  Serial.print(",");
    // Clear the read count
    samplesRead = 0;
  }

  Serial.println();
}

/**
 * Callback function to process the data from the PDM microphone.
 * NOTE: This callback is executed as part of an ISR.
 * Therefore using `Serial` to print messages inside this function isn't supported.
 */
void onPDMdata() {
  // Query the number of available bytes
  int bytesAvailable = PDM.available();

  // Read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // 16-bit, 2 bytes per sample (little-endian)
  samplesRead = bytesAvailable / 2;
}