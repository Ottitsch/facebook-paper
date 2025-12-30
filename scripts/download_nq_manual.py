"""
Download Natural Questions (NQ-Open) manually from HuggingFace.
This bypasses the datasets library glob pattern bug.
"""

import requests
import json
from pathlib import Path
from tqdm import tqdm

# Create data directory
data_dir = Path(__file__).parent.parent / "data" / "nq_open"
data_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Downloading Natural Questions (NQ-Open) from HuggingFace")
print("="*60)
print()

# URLs for NQ-Open dataset files
files_to_download = [
    {
        "name": "NQ-open.dev.jsonl",
        "url": "https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/main/NQ-open.dev.jsonl",
        "description": "Validation set (3,610 samples)"
    },
    {
        "name": "NQ-open.train.jsonl",
        "url": "https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/main/NQ-open.train.jsonl",
        "description": "Training set (~87K samples)"
    }
]

def download_file(url, filepath, description):
    """Download file with progress bar."""
    print(f"Downloading: {filepath.name}")
    print(f"  {description}")
    print(f"  URL: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"  [OK] Downloaded to: {filepath}")
        print()
        return True

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        print()
        return False

# Download files
downloaded_files = []
for file_info in files_to_download:
    filepath = data_dir / file_info["name"]

    # Skip if already exists
    if filepath.exists():
        print(f"[SKIP] {file_info['name']} already exists")
        print(f"  {file_info['description']}")
        print(f"  Size: {filepath.stat().st_size / (1024**2):.2f} MB")
        print()
        downloaded_files.append(filepath)
        continue

    # Download
    success = download_file(file_info["url"], filepath, file_info["description"])
    if success:
        downloaded_files.append(filepath)

# Validate files
print("="*60)
print("Validation")
print("="*60)
print()

for filepath in downloaded_files:
    print(f"Validating: {filepath.name}")

    try:
        # Count lines
        with open(filepath, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)

        # Read first example
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            example = json.loads(first_line)

        print(f"  Samples: {num_lines:,}")
        print(f"  Size: {filepath.stat().st_size / (1024**2):.2f} MB")
        print(f"  Format: {list(example.keys())}")
        print(f"  Sample question: {example['question'][:100]}...")
        print(f"  Sample answer: {example['answer']}")
        print(f"  [OK] Valid JSONL file")
        print()

    except Exception as e:
        print(f"  [FAIL] Validation error: {e}")
        print()

# Summary
print("="*60)
print("Download Complete")
print("="*60)
print()
print(f"Downloaded files saved to: {data_dir}")
print()
print("Files:")
for filepath in downloaded_files:
    if filepath.exists():
        print(f"  - {filepath.name} ({filepath.stat().st_size / (1024**2):.2f} MB)")
print()
print("Next step: Update eval_rag_baseline.py to load from these files")
print("="*60)
