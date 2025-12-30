"""
Download NQ-Open dataset using huggingface_hub library.
This properly downloads dataset files from HuggingFace.
"""

from huggingface_hub import snapshot_download, hf_hub_url, hf_hub_download
from pathlib import Path
import pandas as pd

print("="*60)
print("Downloading NQ-Open Dataset via HuggingFace Hub")
print("="*60)
print()

# Create output directory
data_dir = Path(__file__).parent.parent / "data" / "nq_open"
data_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {data_dir}")
print()

# Method 1: Try to download specific parquet files
print("Attempting to download specific Parquet files...")
print()

repo_id = "google-research-datasets/nq_open"
repo_type = "dataset"

files_to_try = [
    ("nq_open/validation/0000.parquet", "validation"),
    ("nq_open/train/0000.parquet", "train"),
    ("data/validation-00000-of-00001.parquet", "validation"),
    ("data/train-00000-of-00001.parquet", "train"),
]

successful_downloads = []

for filename, split in files_to_try:
    try:
        print(f"Trying: {filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=str(data_dir.parent)
        )
        print(f"  [OK] Downloaded to: {local_path}")

        # Load and inspect
        df = pd.read_parquet(local_path)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        if len(df) > 0:
            print(f"  Sample: {df.iloc[0].to_dict()}")

        successful_downloads.append((split, local_path, len(df)))
        print()

    except Exception as e:
        print(f"  [SKIP] {type(e).__name__}: {str(e)[:100]}")
        print()

# Method 2: If specific files don't work, try snapshot download
if not successful_downloads:
    print("Trying snapshot download (downloads entire dataset repo)...")
    print()

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=str(data_dir.parent),
            allow_patterns=["*.parquet", "*.arrow", "*.json"],  # Only download data files
        )
        print(f"[OK] Dataset downloaded to: {local_dir}")
        print()

        # List downloaded files
        local_path = Path(local_dir)
        parquet_files = list(local_path.rglob("*.parquet"))
        arrow_files = list(local_path.rglob("*.arrow"))

        print(f"Found {len(parquet_files)} Parquet files:")
        for f in parquet_files:
            print(f"  - {f.relative_to(local_path)}")

        print(f"\nFound {len(arrow_files)} Arrow files:")
        for f in arrow_files:
            print(f"  - {f.relative_to(local_path)}")

    except Exception as e:
        print(f"[FAIL] Snapshot download failed: {e}")

# Summary
print()
print("="*60)
print("Summary")
print("="*60)
print()

if successful_downloads:
    print(f"Successfully downloaded {len(successful_downloads)} files:")
    for split, path, num_rows in successful_downloads:
        print(f"  - {split}: {path}")
        print(f"    Rows: {num_rows:,}")
    print()
    print("You can now load these files in eval_rag_baseline.py!")
else:
    print("Could not download specific files.")
    print("Check the snapshot download output above for file locations.")

print("="*60)
