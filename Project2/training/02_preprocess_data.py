"""
Step 2: Preprocess Audio Data for Keyword Spotting
Extracts MFCC features from audio files and prepares training data.
"""

import os
import numpy as np
import tensorflow as tf
import random
from pathlib import Path

# ============================================================
# CONFIGURATION - MODIFY YOUR KEYWORDS HERE
# ============================================================

# Choose 5+ keywords from the dataset
TARGET_KEYWORDS = [
    "go",      # Keyword 1
    "stop",    # Keyword 2
    "up",      # Keyword 3
    "down",    # Keyword 4
    "yes",     # Keyword 5
    "no",      # Keyword 6 (optional extra)
]

# These will be added automatically
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"

# Audio parameters (must match microcontroller settings)
SAMPLE_RATE = 16000          # 16 kHz
AUDIO_LENGTH_SEC = 1.0       # 1 second clips
AUDIO_LENGTH_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH_SEC)

# MFCC parameters
NUM_MFCC = 13                # Number of MFCC coefficients
FRAME_LENGTH = 640           # 40ms at 16kHz
FRAME_STEP = 320             # 20ms at 16kHz (50% overlap)
NUM_FRAMES = 49              # Expected number of frames for 1 second

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Maximum samples per class (to balance dataset)
MAX_SAMPLES_PER_CLASS = 2000

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SPEECH_COMMANDS_DIR = os.path.join(DATA_DIR, "speech_commands")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# ============================================================
# AUDIO LOADING AND MFCC EXTRACTION
# ============================================================

def load_audio_file(file_path):
    """Load a WAV file and return normalized audio samples."""
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # Pad or trim to exact length
    audio_length = tf.shape(audio)[0]
    if audio_length < AUDIO_LENGTH_SAMPLES:
        # Pad with zeros
        padding = AUDIO_LENGTH_SAMPLES - audio_length
        audio = tf.pad(audio, [[0, padding]])
    else:
        # Trim
        audio = audio[:AUDIO_LENGTH_SAMPLES]

    return audio

def extract_mfcc(audio):
    """Extract MFCC features from audio samples."""
    # Compute STFT
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH
    )
    spectrogram = tf.abs(stft)

    # Compute mel spectrogram
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=SAMPLE_RATE / 2
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]
    ))

    # Compute log mel spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCCs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :NUM_MFCC]  # Keep only first N coefficients

    return mfccs

def generate_silence_samples(background_dir, num_samples):
    """Generate silence samples from background noise files."""
    silence_samples = []
    noise_files = list(Path(background_dir).glob("*.wav"))

    if not noise_files:
        print("Warning: No background noise files found. Creating silent samples.")
        for _ in range(num_samples):
            # Pure silence
            audio = np.zeros(AUDIO_LENGTH_SAMPLES, dtype=np.float32)
            mfcc = extract_mfcc(audio).numpy()
            silence_samples.append(mfcc)
        return silence_samples

    for _ in range(num_samples):
        # Pick random noise file
        noise_file = random.choice(noise_files)
        audio_binary = tf.io.read_file(str(noise_file))
        audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
        audio = tf.squeeze(audio, axis=-1).numpy()

        # Pick random 1-second segment
        if len(audio) > AUDIO_LENGTH_SAMPLES:
            start = random.randint(0, len(audio) - AUDIO_LENGTH_SAMPLES)
            audio = audio[start:start + AUDIO_LENGTH_SAMPLES]
        else:
            audio = np.pad(audio, (0, AUDIO_LENGTH_SAMPLES - len(audio)))

        # Scale down to simulate silence (very quiet background)
        audio = audio * 0.1

        mfcc = extract_mfcc(audio).numpy()
        silence_samples.append(mfcc)

    return silence_samples

# ============================================================
# DATA LOADING AND PROCESSING
# ============================================================

def load_keyword_samples(keyword_dir, max_samples=None):
    """Load all samples for a given keyword."""
    samples = []
    wav_files = list(Path(keyword_dir).glob("*.wav"))

    if max_samples:
        wav_files = wav_files[:max_samples]

    for wav_file in wav_files:
        try:
            audio = load_audio_file(str(wav_file))
            mfcc = extract_mfcc(audio).numpy()
            samples.append(mfcc)
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

    return samples

def create_dataset():
    """Create the full dataset with all classes."""

    all_classes = TARGET_KEYWORDS + [SILENCE_LABEL, UNKNOWN_LABEL]
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    print(f"\nClasses ({len(all_classes)} total):")
    for cls, idx in class_to_idx.items():
        print(f"  {idx}: {cls}")

    X_all = []
    y_all = []

    # Load target keywords
    for keyword in TARGET_KEYWORDS:
        keyword_dir = os.path.join(SPEECH_COMMANDS_DIR, keyword)
        if not os.path.exists(keyword_dir):
            print(f"Warning: Directory not found for '{keyword}'")
            continue

        print(f"\nLoading '{keyword}'...")
        samples = load_keyword_samples(keyword_dir, MAX_SAMPLES_PER_CLASS)
        print(f"  Loaded {len(samples)} samples")

        X_all.extend(samples)
        y_all.extend([class_to_idx[keyword]] * len(samples))

    # Generate silence samples
    print(f"\nGenerating '{SILENCE_LABEL}' samples...")
    background_dir = os.path.join(SPEECH_COMMANDS_DIR, "_background_noise_")
    silence_samples = generate_silence_samples(background_dir, MAX_SAMPLES_PER_CLASS)
    print(f"  Generated {len(silence_samples)} samples")
    X_all.extend(silence_samples)
    y_all.extend([class_to_idx[SILENCE_LABEL]] * len(silence_samples))

    # Load unknown samples (other words not in target keywords)
    print(f"\nLoading '{UNKNOWN_LABEL}' samples...")
    unknown_samples = []
    all_dirs = [d for d in os.listdir(SPEECH_COMMANDS_DIR)
                if os.path.isdir(os.path.join(SPEECH_COMMANDS_DIR, d))
                and not d.startswith('_')
                and d not in TARGET_KEYWORDS]

    samples_per_unknown = MAX_SAMPLES_PER_CLASS // len(all_dirs) if all_dirs else 0
    for unknown_word in all_dirs:
        unknown_dir = os.path.join(SPEECH_COMMANDS_DIR, unknown_word)
        samples = load_keyword_samples(unknown_dir, samples_per_unknown)
        unknown_samples.extend(samples)

    random.shuffle(unknown_samples)
    unknown_samples = unknown_samples[:MAX_SAMPLES_PER_CLASS]
    print(f"  Loaded {len(unknown_samples)} samples from {len(all_dirs)} words")
    X_all.extend(unknown_samples)
    y_all.extend([class_to_idx[UNKNOWN_LABEL]] * len(unknown_samples))

    return np.array(X_all), np.array(y_all), all_classes

def split_dataset(X, y):
    """Split dataset into train, validation, and test sets."""
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split
    n_train = int(len(X) * TRAIN_RATIO)
    n_val = int(len(X) * VAL_RATIO)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_dataset(train_data, val_data, test_data, classes):
    """Save processed dataset to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Save class names
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), 'w') as f:
        for cls in classes:
            f.write(cls + '\n')

    print(f"\nDataset saved to {OUTPUT_DIR}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Audio Preprocessing for Keyword Spotting")
    print("=" * 60)

    # Check if dataset exists
    if not os.path.exists(SPEECH_COMMANDS_DIR):
        print(f"\nError: Dataset not found at {SPEECH_COMMANDS_DIR}")
        print("Please run 01_download_dataset.py first!")
        exit(1)

    # Create dataset
    print(f"\nTarget keywords: {TARGET_KEYWORDS}")
    print(f"Audio: {SAMPLE_RATE}Hz, {AUDIO_LENGTH_SEC}s")
    print(f"MFCC: {NUM_FRAMES} frames x {NUM_MFCC} coefficients")

    X, y, classes = create_dataset()
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature shape: {X[0].shape}")

    # Split dataset
    train_data, val_data, test_data = split_dataset(X, y)
    print(f"\nTrain: {len(train_data[0])} samples")
    print(f"Val:   {len(val_data[0])} samples")
    print(f"Test:  {len(test_data[0])} samples")

    # Save
    save_dataset(train_data, val_data, test_data, classes)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("Next step: Run 03_train_model.py")
    print("=" * 60)
