"""
Test Natural Questions dataset loading with fixed pyarrow version.
This validates that pyarrow==13.0.0 resolves the glob pattern errors.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("Natural Questions Dataset Loading Test")
print("="*60)
print()

# Check versions first
print("Checking dependency versions...")
try:
    import pyarrow
    print(f"[OK] pyarrow version: {pyarrow.__version__}")
    if pyarrow.__version__ != "13.0.0":
        print(f"  WARNING: Expected pyarrow 13.0.0, got {pyarrow.__version__}")
except ImportError:
    print("[FAIL] pyarrow not installed!")
    sys.exit(1)

try:
    import datasets
    print(f"[OK] datasets version: {datasets.__version__}")
    if datasets.__version__ != "2.14.0":
        print(f"  WARNING: Expected datasets 2.14.0, got {datasets.__version__}")
except ImportError:
    print("[FAIL] datasets not installed!")
    sys.exit(1)

print()
print("-"*60)
print()

# Try loading Natural Questions with different approaches
from datasets import load_dataset

attempts = [
    {
        "name": "google-research-datasets/nq_open",
        "description": "NQ-Open (PREFERRED - RAG paper uses this)",
        "dataset_name": "google-research-datasets/nq_open",
        "split": "validation"
    },
    {
        "name": "sentence-transformers/natural-questions",
        "description": "Sentence-Transformers preprocessed version",
        "dataset_name": "sentence-transformers/natural-questions",
        "split": "train"  # Note: sentence-transformers version uses 'train' split
    },
    {
        "name": "nq_open",
        "description": "Short name (may not work)",
        "dataset_name": "nq_open",
        "split": "validation"
    }
]

successful_dataset = None

for i, attempt in enumerate(attempts, 1):
    print(f"Attempt {i}/{len(attempts)}: {attempt['name']}")
    print(f"  Description: {attempt['description']}")

    try:
        print(f"  Loading {attempt['dataset_name']} (split={attempt['split']})...")
        dataset = load_dataset(attempt['dataset_name'], split=attempt['split'])

        print(f"  [SUCCESS] Loaded {len(dataset)} examples")
        print(f"  Dataset columns: {dataset.column_names}")

        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n  Sample example:")
            print(f"    Keys: {list(sample.keys())}")

            question = sample.get('question', sample.get('query', 'N/A'))
            answer = sample.get('answer', sample.get('answers', 'N/A'))

            print(f"    Question: {question[:100]}...")
            print(f"    Answer: {answer}")

        # Validate format for RAG evaluation
        if len(dataset) > 0:
            print(f"\n  Validating format for RAG evaluation...")
            sample = dataset[0]

            # Check for question field
            has_question = 'question' in sample or 'query' in sample
            # Check for answer field
            has_answer = 'answer' in sample or 'answers' in sample

            if has_question and has_answer:
                print(f"    [OK] Format is compatible with RAG evaluation")
            else:
                print(f"    ⚠ WARNING: Format may not be compatible")
                print(f"      has_question: {has_question}, has_answer: {has_answer}")

        successful_dataset = {
            'dataset': dataset,
            'name': attempt['name'],
            'dataset_name': attempt['dataset_name'],
            'split': attempt['split']
        }

        print()
        print("="*60)
        print(f"[OK] DATASET LOADED SUCCESSFULLY: {attempt['name']}")
        print("="*60)
        break

    except Exception as e:
        print(f"  [FAIL] FAILED: {type(e).__name__}: {str(e)[:200]}")

    print()
    print("-"*60)
    print()

# Summary
print()
print("="*60)
print("TEST SUMMARY")
print("="*60)

if successful_dataset:
    dataset_info = successful_dataset
    print(f"[OK] SUCCESS")
    print()
    print(f"Dataset: {dataset_info['name']}")
    print(f"  Full name: {dataset_info['dataset_name']}")
    print(f"  Split: {dataset_info['split']}")
    print(f"  Samples: {len(dataset_info['dataset'])}")
    print(f"  Columns: {dataset_info['dataset'].column_names}")
    print()
    print("This dataset can be used for RAG evaluation!")
    print()
    print("Next step: Update eval_rag_baseline.py to use this dataset")
    print(f"  Dataset name: {dataset_info['dataset_name']}")
    print(f"  Split: {dataset_info['split']}")

else:
    print("[FAIL] FAILED")
    print()
    print("None of the Natural Questions datasets could be loaded.")
    print()
    print("Troubleshooting:")
    print("1. Verify pyarrow==13.0.0: pip show pyarrow")
    print("2. Verify datasets==2.14.0: pip show datasets")
    print("3. Clear datasets cache:")
    print("   rm -rf ~/.cache/huggingface/datasets/google-research-datasets___nq_open")
    print("4. Check internet connection")
    print()
    print("Fallback option: Download NQ-Open manually from HuggingFace")

print("="*60)
