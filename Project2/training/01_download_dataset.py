"""
Step 1: Download Google Speech Commands Dataset v2
This script downloads and extracts the dataset for keyword spotting.
"""

import os
import urllib.request
import tarfile
import sys

# Configuration
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_PATH = os.path.join(DATA_DIR, "speech_commands_v0.02.tar.gz")
EXTRACT_PATH = os.path.join(DATA_DIR, "speech_commands")

def download_progress(count, block_size, total_size):
    """Show download progress."""
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()

def download_dataset():
    """Download the Google Speech Commands dataset."""

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(EXTRACT_PATH):
        print(f"Dataset already exists at {EXTRACT_PATH}")
        return EXTRACT_PATH

    # Download
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading dataset from {DATASET_URL}")
        print("This is ~2.3GB and may take a while...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH, download_progress)
        print("\nDownload complete!")
    else:
        print(f"Archive already exists at {DATASET_PATH}")

    # Extract
    print(f"Extracting to {EXTRACT_PATH}...")
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with tarfile.open(DATASET_PATH, 'r:gz') as tar:
        tar.extractall(EXTRACT_PATH)
    print("Extraction complete!")

    return EXTRACT_PATH

def list_available_words(extract_path):
    """List all available words in the dataset."""
    words = []
    for item in os.listdir(extract_path):
        item_path = os.path.join(extract_path, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            words.append(item)
    return sorted(words)

if __name__ == "__main__":
    print("=" * 50)
    print("Google Speech Commands Dataset Downloader")
    print("=" * 50)

    extract_path = download_dataset()

    print("\n" + "=" * 50)
    print("Available words in dataset:")
    print("=" * 50)
    words = list_available_words(extract_path)
    for i, word in enumerate(words, 1):
        print(f"  {i:2}. {word}")

    print(f"\nTotal: {len(words)} words")
    print("\nSpecial folders:")
    print("  - _background_noise_ (for silence/noise samples)")

    print("\n" + "=" * 50)
    print("Next step: Run 02_preprocess_data.py")
    print("=" * 50)
