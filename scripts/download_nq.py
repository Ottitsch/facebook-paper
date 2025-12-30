"""
Download Natural Questions dataset to cache for offline use.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Attempting to download Natural Questions (nq_open) dataset...")
print("This will download to: ~/.cache/huggingface/datasets/nq_open/")
print()

try:
    from datasets import load_dataset

    # Download validation split (3,610 examples)
    print("Downloading validation split...")
    dataset = load_dataset("nq_open", split="validation")
    print(f"✓ Validation split downloaded: {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")

    # Show example
    print("\nExample from dataset:")
    example = dataset[0]
    print(f"  Question: {example['question']}")
    print(f"  Answer: {example['answer']}")

    # Try to access the cache location
    print(f"\nDataset cached successfully!")
    print(f"Cache location: ~/.cache/huggingface/datasets/nq_open/")

except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
    print("\nTrying alternative approach...")

    # Alternative: download directly from HuggingFace Hub
    print("This dataset may need manual download.")
    print("URL: https://huggingface.co/datasets/nq_open")
    sys.exit(1)

print("\nDone!")
