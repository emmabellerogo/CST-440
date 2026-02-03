"""
Step 2: Preprocess Audio Data for Keyword Spotting
Extracts MFCC features from audio files and prepares training data.

CRITICAL: These parameters must match EXACTLY on Arduino!

Optimizations for 90%+ accuracy:
- Data augmentation (time shift, noise, volume)
- Feature normalization
- Balanced classes
"""

import os
import numpy as np
import tensorflow as tf
import random
from pathlib import Path

# ============================================================
# CONFIGURATION - THESE MUST MATCH ARDUINO IMPLEMENTATION!
# ============================================================

# Keywords to detect (6 phonetically distinct words)
TARGET_KEYWORDS = [
    "stop",    # "ah" vowel, hard consonants
    "left",    # short "eh" vowel, "ft" ending
    "right",   # long "ai" diphthong
    "three",   # long "ee" vowel, "th" start
    "cat",     # short "a" vowel
    "bird",    # "er" vowel sound
]

# Special classes (added automatically)
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"

# Audio parameters
SAMPLE_RATE = 16000          # 16 kHz - standard for speech
AUDIO_LENGTH_SEC = 1.0       # 1 second clips
AUDIO_LENGTH_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH_SEC)  # 16000 samples

# MFCC parameters - MUST MATCH ARDUINO!
NUM_MFCC = 13                # Number of MFCC coefficients
FRAME_LENGTH = 640           # 40ms at 16kHz (0.040 * 16000)
FRAME_STEP = 320             # 20ms hop - 50% overlap (0.020 * 16000)
FFT_LENGTH = 1024            # Power of 2 for Arduino compatibility
NUM_MEL_BINS = 40            # Mel filterbank bins
LOWER_FREQ = 20.0            # Lower frequency bound (Hz)
UPPER_FREQ = 8000.0          # Upper frequency bound (Hz) - Nyquist/2

# Calculate expected number of frames
NUM_FRAMES = 1 + (AUDIO_LENGTH_SAMPLES - FRAME_LENGTH) // FRAME_STEP  # = 49

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Maximum samples per class (for balanced dataset)
MAX_SAMPLES_PER_CLASS = 2500  # Increased for better training

# Data augmentation settings
AUGMENTATION_FACTOR = 2      # Multiply training data by this factor
TIME_SHIFT_MAX = 0.1         # Max time shift as fraction of audio length
NOISE_FACTOR_MAX = 0.1       # Max noise amplitude
VOLUME_RANGE = (0.7, 1.3)    # Volume variation range

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SPEECH_COMMANDS_DIR = os.path.join(DATA_DIR, "speech_commands")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# Global background noise for augmentation
BACKGROUND_NOISE = []

# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio_file(file_path):
    """Load a WAV file and return normalized audio samples."""
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)

    # Ensure correct length (pad or trim)
    audio_length = tf.shape(audio)[0]

    if audio_length < AUDIO_LENGTH_SAMPLES:
        # Pad with zeros at the end
        padding = AUDIO_LENGTH_SAMPLES - audio_length
        audio = tf.pad(audio, [[0, padding]])
    else:
        # Trim to exact length
        audio = audio[:AUDIO_LENGTH_SAMPLES]

    return audio.numpy()

# ============================================================
# DATA AUGMENTATION - For better accuracy
# ============================================================

def load_background_noise(background_dir):
    """Load background noise files for augmentation."""
    global BACKGROUND_NOISE
    noise_files = list(Path(background_dir).glob("*.wav"))

    for noise_file in noise_files:
        try:
            audio_binary = tf.io.read_file(str(noise_file))
            audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
            audio = tf.squeeze(audio, axis=-1).numpy()
            BACKGROUND_NOISE.append(audio)
        except:
            continue

    print(f"  Loaded {len(BACKGROUND_NOISE)} background noise files")

def augment_audio(audio):
    """Apply random augmentation to audio."""
    augmented = audio.copy()

    # 1. Time shift (random roll)
    shift = int(random.uniform(-TIME_SHIFT_MAX, TIME_SHIFT_MAX) * len(audio))
    augmented = np.roll(augmented, shift)
    if shift > 0:
        augmented[:shift] = 0
    elif shift < 0:
        augmented[shift:] = 0

    # 2. Add background noise
    if BACKGROUND_NOISE and random.random() > 0.3:
        noise = random.choice(BACKGROUND_NOISE)
        if len(noise) > AUDIO_LENGTH_SAMPLES:
            start = random.randint(0, len(noise) - AUDIO_LENGTH_SAMPLES)
            noise = noise[start:start + AUDIO_LENGTH_SAMPLES]
        else:
            noise = np.pad(noise, (0, AUDIO_LENGTH_SAMPLES - len(noise)))

        noise_factor = random.uniform(0, NOISE_FACTOR_MAX)
        augmented = augmented + noise_factor * noise

    # 3. Volume variation
    volume = random.uniform(*VOLUME_RANGE)
    augmented = augmented * volume

    # Clip to valid range
    augmented = np.clip(augmented, -1.0, 1.0)

    return augmented.astype(np.float32)

# ============================================================
# MFCC EXTRACTION - Must match Arduino implementation!
# ============================================================

def extract_mfcc(audio):
    """
    Extract MFCC features from audio.

    Pipeline:
    1. STFT (Short-Time Fourier Transform)
    2. Magnitude spectrum (NOT power spectrum!)
    3. Mel filterbank
    4. Log compression
    5. DCT (Discrete Cosine Transform) -> MFCCs
    """

    if isinstance(audio, np.ndarray):
        audio = tf.constant(audio, dtype=tf.float32)

    # Compute STFT
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH  # Power of 2 for Arduino compatibility
    )

    # Get magnitude spectrum (not power spectrum!)
    magnitude = tf.abs(stft)

    # Create mel filterbank
    num_spectrogram_bins = magnitude.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_FREQ,
        upper_edge_hertz=UPPER_FREQ
    )

    # Apply mel filterbank
    mel_spectrogram = tf.tensordot(magnitude, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(magnitude.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]
    ))

    # Log compression
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCCs via DCT
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    # Keep only first NUM_MFCC coefficients
    mfccs = mfccs[..., :NUM_MFCC]

    return mfccs.numpy()

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_keyword_samples(keyword_dir, max_samples=None, augment=False):
    """Load and process all audio files for a keyword."""
    samples = []
    wav_files = list(Path(keyword_dir).glob("*.wav"))

    if max_samples:
        random.shuffle(wav_files)
        wav_files = wav_files[:max_samples]

    for wav_file in wav_files:
        try:
            audio = load_audio_file(str(wav_file))

            # Original sample
            mfcc = extract_mfcc(audio)
            if mfcc.shape == (NUM_FRAMES, NUM_MFCC):
                samples.append(mfcc)

            # Augmented samples (only for training)
            if augment:
                for _ in range(AUGMENTATION_FACTOR - 1):
                    aug_audio = augment_audio(audio)
                    aug_mfcc = extract_mfcc(aug_audio)
                    if aug_mfcc.shape == (NUM_FRAMES, NUM_MFCC):
                        samples.append(aug_mfcc)

        except Exception as e:
            print(f"  Error processing {wav_file.name}: {e}")
            continue

    return samples

def generate_silence_samples(background_dir, num_samples):
    """Generate silence samples from background noise files."""
    samples = []
    noise_files = list(Path(background_dir).glob("*.wav"))

    if not noise_files:
        print("  Warning: No background noise files. Creating pure silence.")
        for _ in range(num_samples):
            audio = np.zeros(AUDIO_LENGTH_SAMPLES, dtype=np.float32)
            mfcc = extract_mfcc(audio)
            samples.append(mfcc)
        return samples

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

        # Scale down to simulate quiet background (silence class)
        audio = audio * random.uniform(0.05, 0.2)
        audio = audio.astype(np.float32)

        mfcc = extract_mfcc(audio)

        if mfcc.shape == (NUM_FRAMES, NUM_MFCC):
            samples.append(mfcc)

    return samples

def load_unknown_samples(speech_commands_dir, target_keywords, max_samples, augment=False):
    """Load samples from non-target words for 'unknown' class."""
    samples = []

    # Get all word directories except target keywords and special folders
    all_dirs = [
        d for d in os.listdir(speech_commands_dir)
        if os.path.isdir(os.path.join(speech_commands_dir, d))
        and not d.startswith('_')
        and d not in target_keywords
    ]

    if not all_dirs:
        print("  Warning: No unknown word directories found!")
        return samples

    samples_per_word = max_samples // len(all_dirs)

    for word in all_dirs:
        word_dir = os.path.join(speech_commands_dir, word)
        word_samples = load_keyword_samples(word_dir, samples_per_word, augment=augment)
        samples.extend(word_samples)

    random.shuffle(samples)
    return samples[:max_samples]

# ============================================================
# FEATURE NORMALIZATION - Critical for accuracy!
# ============================================================

def compute_normalization_stats(X_train):
    """Compute mean and std for normalization."""
    # Flatten to (num_samples * num_frames, num_mfcc)
    flat = X_train.reshape(-1, NUM_MFCC)
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std[std < 1e-6] = 1.0  # Avoid division by zero
    return mean, std

def normalize_features(X, mean, std):
    """Normalize features to zero mean, unit variance."""
    return (X - mean) / std

# ============================================================
# MAIN PROCESSING
# ============================================================

def create_dataset():
    """Create the full dataset with all classes."""

    # All classes: keywords + silence + unknown
    all_classes = TARGET_KEYWORDS + [SILENCE_LABEL, UNKNOWN_LABEL]
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    print(f"\nClasses ({len(all_classes)} total):")
    for cls, idx in class_to_idx.items():
        print(f"  {idx}: {cls}")

    # Load background noise for augmentation
    print("\n" + "="*50)
    print("Loading background noise for augmentation...")
    print("="*50)
    background_dir = os.path.join(SPEECH_COMMANDS_DIR, "_background_noise_")
    load_background_noise(background_dir)

    X_all = []
    y_all = []

    # 1. Load target keywords (with augmentation)
    print("\n" + "="*50)
    print("Loading target keywords (with augmentation)...")
    print("="*50)

    for keyword in TARGET_KEYWORDS:
        keyword_dir = os.path.join(SPEECH_COMMANDS_DIR, keyword)

        if not os.path.exists(keyword_dir):
            print(f"  WARNING: Directory not found for '{keyword}'")
            continue

        print(f"\n  Loading '{keyword}'...")
        samples = load_keyword_samples(keyword_dir, MAX_SAMPLES_PER_CLASS, augment=True)
        print(f"    Loaded {len(samples)} samples (with augmentation)")

        X_all.extend(samples)
        y_all.extend([class_to_idx[keyword]] * len(samples))

    # 2. Generate silence samples
    print("\n" + "="*50)
    print("Generating silence samples...")
    print("="*50)

    # Match the augmented size
    target_silence = len(X_all) // len(TARGET_KEYWORDS)
    silence_samples = generate_silence_samples(background_dir, target_silence)
    print(f"  Generated {len(silence_samples)} silence samples")

    X_all.extend(silence_samples)
    y_all.extend([class_to_idx[SILENCE_LABEL]] * len(silence_samples))

    # 3. Load unknown samples (with augmentation)
    print("\n" + "="*50)
    print("Loading unknown word samples (with augmentation)...")
    print("="*50)

    unknown_samples = load_unknown_samples(
        SPEECH_COMMANDS_DIR,
        TARGET_KEYWORDS,
        target_silence,
        augment=True
    )
    print(f"  Loaded {len(unknown_samples)} unknown samples")

    X_all.extend(unknown_samples)
    y_all.extend([class_to_idx[UNKNOWN_LABEL]] * len(unknown_samples))

    return np.array(X_all, dtype=np.float32), np.array(y_all), all_classes

def split_dataset(X, y):
    """Split into train, validation, and test sets."""

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Calculate split points
    n_train = int(len(X) * TRAIN_RATIO)
    n_val = int(len(X) * VAL_RATIO)

    # Split
    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_dataset(train_data, val_data, test_data, classes, norm_stats):
    """Save processed dataset to disk."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    mean, std = norm_stats

    # Save numpy arrays
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Save normalization stats (needed for Arduino!)
    np.save(os.path.join(OUTPUT_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "norm_std.npy"), std)

    # Save class names
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), 'w') as f:
        for cls in classes:
            f.write(cls + '\n')

    # Save MFCC parameters for reference (MUST MATCH ARDUINO!)
    with open(os.path.join(OUTPUT_DIR, "mfcc_params.txt"), 'w') as f:
        f.write(f"SAMPLE_RATE={SAMPLE_RATE}\n")
        f.write(f"AUDIO_LENGTH_SAMPLES={AUDIO_LENGTH_SAMPLES}\n")
        f.write(f"NUM_MFCC={NUM_MFCC}\n")
        f.write(f"FRAME_LENGTH={FRAME_LENGTH}\n")
        f.write(f"FRAME_STEP={FRAME_STEP}\n")
        f.write(f"FFT_LENGTH={FFT_LENGTH}\n")
        f.write(f"NUM_FRAMES={NUM_FRAMES}\n")
        f.write(f"NUM_MEL_BINS={NUM_MEL_BINS}\n")
        f.write(f"LOWER_FREQ={LOWER_FREQ}\n")
        f.write(f"UPPER_FREQ={UPPER_FREQ}\n")
        f.write(f"\n# Normalization stats (apply on Arduino too!)\n")
        f.write(f"NORM_MEAN={mean.tolist()}\n")
        f.write(f"NORM_STD={std.tolist()}\n")

    print(f"\nDataset saved to: {OUTPUT_DIR}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("STEP 2: Preprocessing Audio Data")
    print("Optimized for 90%+ accuracy")
    print("="*60)

    # Check dataset exists
    if not os.path.exists(SPEECH_COMMANDS_DIR):
        print(f"\nERROR: Dataset not found at {SPEECH_COMMANDS_DIR}")
        print("Please ensure the speech_commands folder exists.")
        exit(1)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Target keywords: {TARGET_KEYWORDS}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Audio length: {AUDIO_LENGTH_SEC}s ({AUDIO_LENGTH_SAMPLES} samples)")
    print(f"  MFCC shape: {NUM_FRAMES} frames x {NUM_MFCC} coefficients")
    print(f"  Frame length: {FRAME_LENGTH} samples ({FRAME_LENGTH/SAMPLE_RATE*1000:.1f}ms)")
    print(f"  Frame step: {FRAME_STEP} samples ({FRAME_STEP/SAMPLE_RATE*1000:.1f}ms)")
    print(f"  FFT length: {FFT_LENGTH} (power of 2 for Arduino)")
    print(f"  Augmentation factor: {AUGMENTATION_FACTOR}x")

    # Create dataset
    X, y, classes = create_dataset()

    print("\n" + "="*60)
    print("Dataset Summary (before normalization)")
    print("="*60)
    print(f"  Total samples: {len(X)}")
    print(f"  Feature shape: {X[0].shape}")
    print(f"  Classes: {len(classes)}")

    # Print class distribution
    print("\n  Class distribution:")
    for i, cls in enumerate(classes):
        count = np.sum(y == i)
        print(f"    {cls}: {count} samples")

    # Split dataset
    train_data, val_data, test_data = split_dataset(X, y)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Compute normalization stats from training data
    print("\n" + "="*60)
    print("Computing normalization statistics...")
    print("="*60)
    mean, std = compute_normalization_stats(X_train)
    print(f"  Mean per MFCC: {mean}")
    print(f"  Std per MFCC: {std}")

    # Normalize all sets
    print("\n  Normalizing features...")
    X_train = normalize_features(X_train, mean, std)
    X_val = normalize_features(X_val, mean, std)
    X_test = normalize_features(X_test, mean, std)

    # Save
    save_dataset(
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        classes,
        (mean, std)
    )

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print("\nOptimizations applied:")
    print("  [x] Data augmentation (time shift, noise, volume)")
    print("  [x] Feature normalization (zero mean, unit variance)")
    print("  [x] Balanced classes")
    print("  [x] FFT length = 1024 (Arduino compatible)")
    print("\nNext: Run 02_train_model.py")
