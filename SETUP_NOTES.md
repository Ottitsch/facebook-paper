# Setup Notes - RAG Evaluation Environment

## Environment Resolution Journey

### Problem: Python 3.13 Incompatibility

**Root Cause**: Python 3.13 is too new for RAG dependencies:
- pyarrow 18.0.0+ (only version for Python 3.13) removed `PyExtensionType` API
- datasets 2.14.0 (required for wiki_dpr) expects old pyarrow API with `PyExtensionType`
- Creates circular dependency impossible to resolve with Python 3.13

**Solution**: ✅ Use Python 3.11

### Critical Dependency Chain

⚠️ **THESE VERSIONS ARE NON-NEGOTIABLE:**

```
Python 3.11.9
└── numpy 1.26.4 (MUST be <2.0.0)
    └── pyarrow 13.0.0 ⚠️ MUST BE EXACTLY 13.0.0
        └── datasets 2.14.0 (ONLY version with wiki_dpr support)
            └── transformers 4.35.0+
                └── RAG models work correctly
```

**Why pyarrow 13.0.0 specifically:**
- pyarrow 14.x, 15.x, 22.x: Causes `PyExtensionType` errors
- pyarrow <13.0.0: Missing features needed by datasets
- pyarrow ==13.0.0: ✓ ONLY version that works

**Why datasets 2.14.0 specifically:**
- datasets 2.20.0+: Has LocalFileSystem glob pattern bugs on Windows
- datasets <2.14.0: Incompatible with wiki_dpr dataset
- datasets ==2.14.0: ✓ Last version before breaking changes

## Final Working Configuration

**Environment**: `venv311` (Python 3.11.9)

**Key Dependencies:**
```
torch==2.7.1+cu118       # CUDA 11.8 support
transformers==4.57.3     # Latest HF transformers
datasets==2.14.0         # CRITICAL - exact version
pyarrow==13.0.0          # CRITICAL - exact version
numpy==1.26.4            # <2.0.0 for pyarrow compatibility
faiss-cpu==1.13.2        # Vector similarity search
bitsandbytes==0.49.0     # 4-bit quantization
accelerate==1.12.0       # Model optimization
fsspec==2024.5.0         # File system operations
huggingface_hub>=0.19.0  # Dataset downloads
pandas>=2.0.0            # Data manipulation
```

## Complete Setup Instructions

### 1. Create Python 3.11 Environment

```bash
# Check Python 3.11 is available
py -0  # List all Python versions

# Create virtual environment
py -3.11 -m venv venv311

# Activate (Windows)
venv311\Scripts\activate

# Activate (Linux/Mac)
source venv311/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower)
pip install torch torchvision torchaudio
```

### 4. Install All Requirements

```bash
# Install from requirements.txt
pip install -r requirements.txt

# This will install all dependencies with correct versions
```

### 5. Verify Critical Versions

```bash
# Check pyarrow (MUST be 13.0.0)
python -c "import pyarrow; print(f'pyarrow: {pyarrow.__version__}')"

# Check datasets (MUST be 2.14.0)
python -c "import datasets; print(f'datasets: {datasets.__version__}')"

# Check numpy (MUST be <2.0.0)
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

# Check torch CUDA
python -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
pyarrow: 13.0.0
datasets: 2.14.0
numpy: 1.26.4
torch: 2.7.1+cu118
CUDA available: True
```

### 6. Download Natural Questions Dataset

```bash
# Method 1: Using HuggingFace Hub (recommended)
python scripts/download_nq_hf_hub.py

# Method 2: Direct Parquet download
python scripts/download_nq_hub.py

# Method 3: Manual JSONL download
python scripts/download_nq_manual.py
```

This creates: `data/datasets--google-research-datasets--nq_open/`

### 7. Run Test Evaluation

```bash
# Quick test (10 samples, ~2 min)
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 10

# Should see:
# - Model loading successfully
# - Wikipedia index loading (downloads ~36GB on first run)
# - Evaluation running with progress bar
# - Results printed at end
```

## First Run Expectations

**Wikipedia Index Download:**
- Size: ~36GB compressed
- Files: 157 data files
- Time: 30-60 minutes (depends on internet speed)
- Location: `~/.cache/huggingface/datasets/wiki_dpr/`
- Contains: 21,015,300 Wikipedia passages
- **Only happens once** - then cached locally

**Model Downloads:**
- `facebook/rag-sequence-nq`: ~2GB
- Includes: DPR encoders + BART generator weights
- Location: `~/.cache/huggingface/hub/`
- Time: 5-10 minutes

**Total First Run:**
- Download time: ~40-70 minutes
- Disk space needed: ~40GB
- Subsequent runs: Start immediately (uses cache)

## Hardware Requirements

**Minimum:**
- RAM: 32GB (Wikipedia index needs ~36GB when loaded)
- GPU VRAM: 6GB (for FP16 inference)
- Disk Space: 40GB free
- CPU: Modern multi-core (8+ cores recommended)

**Tested Configuration (Working):**
- RAM: 48GB ✓
- GPU: RTX 2080 Super (8GB VRAM) ✓
- CPU: AMD Ryzen 9 3900X (12-core) ✓
- OS: Windows 11 ✓

**Performance:**
- With FP16: ~0.09 questions/second (batch_size=4)
- With FP32: ~0.06 questions/second (batch_size=4)
- CPU only: ~0.01 questions/second (very slow)


# Running Experiments

### Baseline RAG-BART Evaluation

**Basic evaluation:**
```bash
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 100 --batch_size 4
```

**Save results:**
```bash
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 100 --output_file results/metrics/baseline_100.json
```

**Adjust for GPU memory:**
```bash
# Low VRAM (4-6GB)
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 100 --batch_size 1 --no_fp16

# Medium VRAM (8GB) - default
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 100 --batch_size 4

# High VRAM (12GB+)
venv311\Scripts\python.exe experiments\eval_rag_baseline.py --max_samples 100 --batch_size 8
```

### Flan-T5 Variant Evaluations

**Flan-T5-small (fastest, 77M params):**
```bash
venv311\Scripts\python.exe experiments\eval_rag_flan_t5.py --model_name google/flan-t5-small --max_samples 100 --output_file results/metrics/flan_t5_small_100.json
```

**Flan-T5-base (best balance, 248M params):**
```bash
venv311\Scripts\python.exe experiments\eval_rag_flan_t5.py --model_name google/flan-t5-base --max_samples 100 --output_file results/metrics/flan_t5_base_100.json
```

**Flan-T5-large with 4-bit quantization (best accuracy, 494M params):**
```bash
venv311\Scripts\python.exe experiments\eval_rag_flan_t5.py --model_name google/flan-t5-large --max_samples 100 --use_4bit --output_file results/metrics/flan_t5_large_4bit_100.json
```

**Quick test (10 samples) for any variant:**
```bash
venv311\Scripts\python.exe experiments\eval_rag_flan_t5.py --model_name google/flan-t5-base --max_samples 10
```

### Analysis and Visualization

**Generate comparison table:**
```bash
venv311\Scripts\python.exe analysis\compare_results.py
```

**Generate all plots:**
```bash
venv311\Scripts\python.exe analysis\visualize_results.py
```

**Outputs:**
- `results/comparison_table.csv` - Detailed metrics comparison
- `results/figures/*.png` - 5 visualization plots (300 DPI)




## Import Gotchas

RAG classes MUST be imported directly from module files (not via lazy-loaded `__init__.py`):

```python
# ✅ CORRECT imports
from transformers.models.rag.modeling_rag import RagSequenceForGeneration
from transformers.models.rag.tokenization_rag import RagTokenizer
from transformers.models.rag.retrieval_rag import RagRetriever

# ❌ WRONG imports (will fail)
from transformers import RagSequenceForGeneration  # Lazy loading breaks
from transformers import RagTokenizer              # Lazy loading breaks
from transformers import RagRetriever              # Lazy loading breaks
```

## Issues Encountered & Solutions

### Issue 1: Dataset Loading Failures

**Problem**: `datasets.load_dataset("google-research-datasets/nq_open")` fails with:
- `NotImplementedError: Loading a dataset cached in a LocalFileSystem`
- Glob pattern errors on Windows

**Root Cause**: datasets 2.20.0+ has bug with Windows file system paths

**Solution**:
1. Use datasets==2.14.0 (last working version)
2. Download Parquet files via `huggingface_hub` library
3. Load directly from Parquet files in `eval_rag_baseline.py`

### Issue 2: PyArrow Version Conflicts

**Problem**:
- `module 'pyarrow' has no attribute 'PyExtensionType'`
- Complete failure to load any datasets

**Root Cause**: pyarrow 14.x+ removed PyExtensionType API

**Solution**:
```bash
pip install pyarrow==13.0.0  # Exactly 13.0.0
```

### Issue 3: Evaluation Performance Below Expected

**Problem**: 100-sample test shows 27% EM (expected 40-45%)

**Possible Causes**:
1. Small sample variance
2. Answer normalization differences
3. Metric computation mismatch with paper
4. Generation parameters differ from paper defaults

**Investigation Steps**:
1. Run larger sample sizes (200, 500) for stable estimate
2. Manually inspect example predictions
3. Compare metric implementation with paper
4. Check generation parameters (beam search, max_length, etc.)

### Issue 4: Wikipedia Index Location

**Problem**: Downloaded 74GB wiki_dpr data, hard to find

**Current Location**: `C:\Users\<username>\.cache\huggingface\datasets\wiki_dpr\`
- Contains: 161 .arrow files
- Total passages: 21,015,300
- Successfully loading via RagRetriever

**Note**: This is automatic - no manual management needed

## Dataset Loading Workaround Details

Our evaluation script uses a custom loader to bypass datasets library bugs:

**File**: `experiments/eval_rag_baseline.py` (lines 84-150)

**Method**:
1. Locate Parquet files in cache: `data/datasets--google-research-datasets--nq_open/`
2. Find snapshot directory (has version hash)
3. Load directly with pandas: `pd.read_parquet()`
4. Convert to list of dicts for evaluation

**Advantages**:
- Bypasses datasets library completely for NQ loading
- Works reliably on Windows
- Faster loading (no dataset library overhead)
- Still uses datasets library for wiki_dpr (via RagRetriever)

## Troubleshooting Commands

**Reset pyarrow if broken:**
```bash
pip uninstall pyarrow -y
pip install pyarrow==13.0.0
```

**Reset datasets if broken:**
```bash
pip uninstall datasets -y
pip install datasets==2.14.0
```

**Check installed versions:**
```bash
pip list | grep -E "torch|transformers|datasets|pyarrow|numpy|faiss"
```

**Clear HuggingFace cache (if corrupted):**
```bash
# WARNING: This deletes downloaded models and datasets
rm -rf ~/.cache/huggingface/  # Linux/Mac
rmdir /s ~/.cache/huggingface/  # Windows
```

