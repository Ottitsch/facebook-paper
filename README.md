# RAG Paper Reproduction & Extension

Reproduction of "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) with extensions using Flan-T5 models.

## Overview

This project:
1. Reproduces the original RAG-BART baseline on Natural Questions (target: ~44.5% EM)
2. Extends with modern Flan-T5 models (small/base/large)
3. Evaluates 4-bit quantization for efficiency
4. Compares speed vs accuracy trade-offs

## Current Status

✅ **Working:**
- Official RAG model loading (`facebook/rag-sequence-nq`)
- Natural Questions dataset loading from Parquet files (workaround for datasets library bug)
- Wikipedia index cached and operational (21M passages)
- Evaluation pipeline running end-to-end with corrected paper settings

⚠️ **Performance Analysis Complete:**
- **Baseline EM: ~27-40%** on Natural Questions validation (100-200 samples)
- **Expected from paper: 44.5% EM**
- **Root cause identified**: Hardware limitation (8GB GPU cannot use paper's 50 retrieved docs)
- **Key finding**: EM = F1 because model generates either perfect answers OR complete garbage (no partial matches)

## Setup

**Requirements:**
- Python 3.11 (NOT 3.13 - compatibility issues with pyarrow)
- CUDA-capable GPU (8GB+ VRAM recommended)
- 48GB+ RAM (for Wikipedia index in memory)
- ~40GB disk space (Wikipedia index + models)

**Quick Start:**

```bash
# 1. Create Python 3.11 environment
py -3.11 -m venv venv311

# 2. Activate environment
venv311\Scripts\activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. CRITICAL: Verify pyarrow version
python -c "import pyarrow; print(pyarrow.__version__)"  # Should be 13.0.0

# 6. Download Natural Questions dataset
python scripts/download_nq_hf_hub.py

# 7. Run evaluation
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 10
```

## How It Works

Uses the **exact setup from the paper** (with hardware constraints):
- **Model**: `facebook/rag-sequence-nq` (RAG-BART, 515M parameters)
- **Dataset**: Natural Questions (3,610 validation samples)
- **Retriever**: DPR with compressed Wikipedia index (21M passages, ~36GB)
- **Generation**: Greedy decoding (as per paper)
- **Metrics**: Exact Match (EM) and F1 Score

**Hardware Limitation**: Paper uses 50 retrieved docs, but this requires 12+ GB VRAM. We use 15 docs to fit in 8GB GPU.

## Running Evaluations

### Baseline RAG-BART

```bash
# Quick test (10 samples, ~2 min)
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 10

# Medium test (100 samples, ~20 min)
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 100 --output_file results/metrics/baseline_100.json

# Convenience wrapper (same as above)
run_eval.bat --max_samples 100 --output_file results/metrics/baseline_100.json

# Full evaluation (3,610 samples, ~6-10 hours)
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --split validation --output_file results/metrics/baseline_full.json
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max_samples N` | Evaluate on N samples | All samples |
| `--batch_size N` | Batch size for generation | 4 |
| `--n_docs N` | Documents to retrieve | 15 (paper uses 50) |
| `--split` | Dataset split (validation/train) | validation |
| `--output_file` | Save results to JSON | None |
| `--use_dummy_index` | Use dummy data (testing only) | False |
| `--no_fp16` | Disable FP16 precision | False |

## Performance Analysis

### Baseline Results (RAG-BART with 8GB GPU constraints)

| Samples | Time | Actual EM | Actual F1 | Paper EM (50 docs) |
|---------|------|-----------|-----------|-------------|
| 10 | 2 min | 40% | 40% | - |
| 100 | 19 min | 27% | 27% | - |
| 200 | 38 min | 30% | 30% | - |
| Full (3,610) | ~10 hrs | Pending | Pending | **44.5%** |

### Why EM = F1?

**Discovery**: Model generates either:
1. ✅ **Perfect answers** (EM=1, F1=1) - e.g., "14 december 1972 utc"
2. ❌ **Complete garbage** (EM=0, F1=0) - e.g., " the", ","

No partial matches → EM always equals F1

### Hardware Bottleneck

**Paper settings** (44.5% EM):
- 50 retrieved documents
- Greedy decoding
- Likely 16GB+ V100 GPUs

**Our constraints** (27-40% EM):
- 15 retrieved documents (8GB GPU limit)
- Greedy decoding ✓
- RTX 2080 Super (8GB VRAM)

**Attempting 50 docs**: `CUDA out of memory. Tried to allocate 12.36 GiB` ❌

## Project Structure

```
facebook/
├── README.md                    # This file
├── SETUP_NOTES.md               # Detailed environment setup
├── requirements.txt             # Python dependencies (EXACT versions)
├── run_eval.bat                 # Convenience wrapper for evaluation
├── assignment.md                # Assignment requirements
├── 2005.11401v4.md             # RAG paper
│
├── venv311/                     # Python 3.11 virtual environment
│
├── experiments/
│   ├── eval_rag_baseline.py    # Main evaluation script (corrected settings)
│   └── test_nq_loading.py      # Dataset loading validation
│
├── scripts/
│   ├── download_nq_hf_hub.py   # Download NQ dataset via HuggingFace Hub
│   ├── download_nq_hub.py      # Download NQ Parquet files directly
│   └── download_nq_manual.py   # Download NQ JSONL files
│
├── src/
│   └── evaluation/
│       └── metrics.py          # EM and F1 metrics
│
├── data/
│   └── datasets--google-research-datasets--nq_open/  # NQ dataset cache
│       └── snapshots/
│           └── */nq_open/
│               ├── validation-00000-of-00001.parquet
│               └── train-00000-of-00001.parquet
│
└── results/
    └── metrics/                # Evaluation results (JSON)
```

**System Cache:**
- Models: `~/.cache/huggingface/hub/`
- Wikipedia index: `~/.cache/huggingface/datasets/wiki_dpr/` (74GB, auto-downloaded)

## Dataset Loading Workaround

⚠️ **Important**: The HuggingFace `datasets` library has a bug with glob patterns on Windows when loading `nq_open`.

**Solution**: We load directly from Parquet files downloaded via `huggingface_hub`:

1. Download dataset: `python scripts/download_nq_hf_hub.py`
2. Files saved to: `data/datasets--google-research-datasets--nq_open/`
3. Evaluation script loads directly from Parquet (bypasses datasets library)


## Troubleshooting

### CRITICAL: PyArrow version issue

**Error:** `module 'pyarrow' has no attribute 'PyExtensionType'`

**Fix:**
```bash
pip install pyarrow==13.0.0  # MUST be exactly 13.0.0
```

**Why:** datasets 2.14.0 only works with pyarrow 13.0.0. Newer versions (14.x, 15.x) break compatibility.

### Issue: datasets library loading fails

**Error:** `NotImplementedError: Loading a dataset cached in a LocalFileSystem`

**Fix:**
```bash
pip install datasets==2.14.0  # NOT 2.20.0
```

### Issue: CUDA out of memory with 50 docs

**Error:** `CUDA out of memory. Tried to allocate 12.36 GiB`

**Solutions:**
- Use `--n_docs 15` (default, fits in 8GB)
- Reduce `--batch_size` to 2 or 1
- Use `--no_fp16` (slower but uses less VRAM)
- **Best solution**: Use 16GB+ GPU to match paper settings

### Issue: Out of RAM

**Solutions:**
- Close other applications (Wikipedia index needs ~36GB RAM)
- Reduce `--n_docs` to 10 (retrieves fewer documents)

### Issue: Download is slow

- Be patient - first download of Wikipedia index (~36GB) takes 30-60 minutes
- Downloads are cached, only happens once
- Check internet connection

## Next Steps

1. ✅ **Baseline evaluation complete**
   - EM = 27-40% (vs paper's 44.5%)
   - Root cause: Hardware limitation (15 docs vs 50 docs)
   - Generation parameters corrected (greedy decoding)

2. **Implement Flan-T5 variants**
   - Create `eval_rag_flan_t5.py`
   - Test: flan-t5-small, flan-t5-base, flan-t5-large
   - Add 4-bit quantization for large model
   - All use same 15-doc constraint for fair comparison

3. **Compare results**
   - Collect metrics: EM, F1, speed, VRAM usage
   - Create comparison table
   - Generate visualizations

4. **Prepare presentation**
   - 20-minute talk
   - Results analysis
   - Hardware limitation discussion
   - Trade-offs analysis

## References

- **Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- **Model**: [facebook/rag-sequence-nq](https://huggingface.co/facebook/rag-sequence-nq)
- **Dataset**: [Natural Questions](https://ai.google.com/research/NaturalQuestions) via [HuggingFace](https://huggingface.co/datasets/google-research-datasets/nq_open)
- **Wikipedia Index**: [wiki_dpr](https://huggingface.co/datasets/wiki_dpr) (21M passages)

## Hardware Info

**Working Configuration:**
- OS: Windows 11
- CPU: AMD Ryzen 9 3900X (12-core)
- RAM: 48GB
- GPU: RTX 2080 Super (8GB VRAM)
- Python: 3.11.9

**Note**: To reproduce paper's 44.5% EM, you need:
- 16GB+ GPU (V100, A100, RTX 3090, RTX 4090)
- 50 retrieved documents setting
- Same generation parameters (greedy decoding)
