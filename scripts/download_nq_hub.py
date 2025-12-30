"""
Download Natural Questions using HuggingFace Hub API.
Bypasses datasets library to avoid glob pattern bug.
"""

import pandas as pd
from pathlib import Path

print("="*60)
print("Downloading NQ-Open using direct Parquet access")
print("="*60)
print()

# Create output directory
data_dir = Path(__file__).parent.parent / "data" / "nq_open"
data_dir.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset Parquet URLs (standard pattern for auto-converted datasets)
# Format: https://huggingface.co/datasets/{org}/{dataset}/resolve/refs/convert/parquet/{dataset}/{split}/{file}.parquet

base_url = "https://huggingface.co/datasets/google-research-datasets/nq_open/resolve/refs/convert/parquet/nq_open"

splits_to_download = [
    {
        "name": "validation",
        "url": f"{base_url}/validation/0000.parquet",
        "output_file": "nq_open_validation.parquet",
        "description": "Validation set (3,610 samples)"
    },
    {
        "name": "train",
        "url": f"{base_url}/train/0000.parquet",
        "output_file": "nq_open_train.parquet",
        "description": "Training set (~87K samples)"
    }
]

def download_and_save_parquet(url, output_path, description):
    """Download Parquet file and save locally."""
    print(f"Downloading: {output_path.name}")
    print(f"  {description}")
    print(f"  URL: {url}")

    try:
        # Read directly from URL using pandas
        df = pd.read_parquet(url)

        # Save to local file
        df.to_parquet(output_path, index=False)

        print(f"  [OK] Downloaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")

        # Show sample
        if len(df) > 0:
            print(f"  Sample question: {df['question'].iloc[0][:100]}...")
            print(f"  Sample answer: {df['answer'].iloc[0]}")

        print(f"  Saved to: {output_path}")
        print()
        return True, len(df)

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        print()
        return False, 0

# Download files
total_samples = 0
successful_downloads = []

for split_info in splits_to_download:
    output_path = data_dir / split_info["output_file"]

    # Skip if already exists
    if output_path.exists():
        print(f"[SKIP] {split_info['output_file']} already exists")
        print(f"  Size: {output_path.stat().st_size / (1024**2):.2f} MB")

        # Load to count rows
        try:
            df = pd.read_parquet(output_path)
            print(f"  Rows: {len(df)}")
            total_samples += len(df)
            successful_downloads.append(output_path)
        except:
            pass

        print()
        continue

    # Download
    success, num_rows = download_and_save_parquet(
        split_info["url"],
        output_path,
        split_info["description"]
    )

    if success:
        total_samples += num_rows
        successful_downloads.append(output_path)

# Summary
print("="*60)
print("Download Summary")
print("="*60)
print()
print(f"Successfully downloaded: {len(successful_downloads)} files")
print(f"Total samples: {total_samples:,}")
print()
print(f"Files saved to: {data_dir}")
for filepath in successful_downloads:
    if filepath.exists():
        print(f"  - {filepath.name} ({filepath.stat().st_size / (1024**2):.2f} MB)")
print()
print("Next step: Update eval_rag_baseline.py to load from these Parquet files")
print("="*60)
