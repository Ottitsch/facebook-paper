# RAG Paper Reproduction & Extension

Reproduction of "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) with extensions using Flan-T5 models, reranking strategies, and RAG Fusion.

## Overview

This project:
1. Reproduces the original RAG-BART baseline on Natural Questions
2. Extends with modern Flan-T5 models (small/base/large)
3. Implements three reranking strategies (basic, enhanced, diversity)
4. Explores RAG Fusion with query variations and Reciprocal Rank Fusion
5. Evaluates 4-bit quantization for efficiency
6. Compares speed vs accuracy trade-offs

## Quick Links

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Executive summary with complete deliverables
- [SETUP_NOTES.md](SETUP_NOTES.md) - Comprehensive environment setup guide
- [RERANKER_ANALYSIS_REPORT.md](RERANKER_ANALYSIS_REPORT.md) - Detailed reranking strategy analysis

---

## Results Summary

### Main Results (100 samples)

| Model | Params | EM | F1 | Speed (q/s) | Speedup |
|-------|--------|-----|-----|-------------|---------|
| **RAG-BART (baseline)** | 515M | 27.0% | 27.0% | 0.088 | 1.0x |
| Flan-T5-small | 77M | 13.0% | 19.7% | 5.02 | 57x |
| **Flan-T5-base** | 248M | 26.0% | 31.9% | 1.87 | **21x** |
| Flan-T5-large (4-bit) | 494M | 26.0% | 34.1% | 1.20 | 14x |

### Reranking Strategies (Flan-T5-base, 100 samples)

| Strategy | EM | F1 | Speed (q/s) | Notes |
|----------|-----|-----|-------------|-------|
| Basic | 21.0% | 30.6% | 0.99 | Cross-encoder only |
| **Enhanced** | **26.0%** | **33.0%** | **0.96** | CE + TF-IDF + coverage |
| Diversity | 24.0% | 34.3% | 0.62 | MMR-style diversity |

### RAG Fusion (Flan-T5-base)

| Samples | EM | F1 | Speed (q/s) |
|---------|-----|-----|-------------|
| 100 | 26.0% | 32.8% | 0.71 |
| 3,610 (full) | 22.8% | 30.3% | 0.72 |

### Key Findings

1. **Flan-T5-base: Best Balance** - Matches baseline accuracy (26% vs 27% EM) while being 21x faster
2. **Enhanced Reranker: Production Ready** - Best accuracy with reasonable speed (26% EM, 33% F1)
3. **Diversity Reranker: Best F1** - Highest F1 score (34.3%) for comprehensive answers
4. **F1 > EM for Flan-T5** - Unlike baseline where F1 = EM, Flan-T5 generates partial matches
5. **4-bit Quantization Works** - Flan-T5-large maintains 26% EM with 75% memory reduction

---

## Setup

### Requirements
- Python 3.11 (NOT 3.13 - compatibility issues)
- CUDA-capable GPU (8GB+ VRAM recommended)
- 48GB+ RAM (for Wikipedia index)
- ~40GB disk space

### Quick Start

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

# 7. Run quick test
python experiments/eval_rag_baseline.py --max_samples 10
```

---

## Running Evaluations

### Baseline RAG-BART

```bash
# Quick test (10 samples)
python experiments/eval_rag_baseline.py --max_samples 10

# Standard evaluation (100 samples)
python experiments/eval_rag_baseline.py --max_samples 100 --output_file results/metrics/baseline_100.json
```

### Flan-T5 Variants

```bash
# Flan-T5-small (fastest)
python experiments/eval_rag_flan_t5.py --model_name google/flan-t5-small --max_samples 100 --output_file results/metrics/flan_t5_small_100.json

# Flan-T5-base (best balance)
python experiments/eval_rag_flan_t5.py --model_name google/flan-t5-base --max_samples 100 --output_file results/metrics/flan_t5_base_100.json

# Flan-T5-large with 4-bit quantization
python experiments/eval_rag_flan_t5.py --model_name google/flan-t5-large --max_samples 100 --use_4bit --output_file results/metrics/flan_t5_large_4bit_100.json
```

### Reranking Strategies

```bash
# Run all three reranking strategies
python experiments/eval_rag_flan_t5_reranked.py --max_samples 100 --output_file results/metrics/reranker_strategies_100.json
```

### RAG Fusion

```bash
# RAG Fusion with query variations
python experiments/eval_rag_fusion.py --max_samples 100 --output_file results/metrics/rag_fusion_100.json

# Full validation set
python experiments/eval_rag_fusion.py --output_file results/metrics/rag_fusion_results.json
```

### Analysis & Visualization

```bash
# Generate comparison table
python analysis/compare_results.py

# Generate all visualization plots
python analysis/visualize_results.py
```

---

## Project Structure

```
facebook-paper/
├── README.md                           # This file
├── PROJECT_SUMMARY.md                  # Executive summary
├── SETUP_NOTES.md                      # Environment setup guide
├── RERANKER_ANALYSIS_REPORT.md         # Reranker strategy analysis
├── requirements.txt                    # Dependencies (exact versions)
│
├── experiments/
│   ├── eval_rag_baseline.py           # RAG-BART baseline
│   ├── eval_rag_flan_t5.py            # Flan-T5 variants
│   ├── eval_rag_flan_t5_reranked.py   # Flan-T5 with reranking
│   ├── eval_rag_fusion.py             # RAG Fusion (query variations + RRF)
│   ├── eval_rag_t5_promt.py           # T5 prompt engineering
│   ├── eval_rag_custom.py             # Custom RAG implementation
│   ├── compare_rag_variants.py         # RAG variant comparison
│   ├── reranker.py                     # Reranker strategies
│   ├── prompts.py                      # Prompt templates
│   └── test_nq_loading.py             # Dataset loading validation
│
├── src/
│   └── evaluation/
│       └── metrics.py                  # EM and F1 computation
│
├── analysis/
│   ├── compare_results.py              # Comparison tables
│   └── visualize_results.py            # Visualization plots
│
├── scripts/
│   └── download_nq_hf_hub.py          # Dataset download
│
├── presentation/
│   └── PRESENTATION_OUTLINE.md         # 20-minute talk outline
│
├── data/                               # Dataset cache
│   └── datasets--google-research-datasets--nq_open/
│
└── results/
    ├── metrics/                        # JSON result files
    │   ├── baseline_100samples.json
    │   ├── baseline_200samples.json
    │   ├── flan_t5_small_100.json
    │   ├── flan_t5_base_100.json
    │   ├── flan_t5_large_4bit_100.json
    │   ├── reranker_strategies_100.json
    │   ├── rag_fusion_100.json
    │   ├── rag_fusion_results.json     # Full 3,610 samples
    │   └── t5_prompt_100.json
    └── figures/                        # Visualization plots (300 DPI)
        ├── accuracy_comparison.png
        ├── speed_vs_accuracy.png
        ├── speedup_comparison.png
        ├── f1_em_gap.png
        └── combined_summary.png
```

---

## Architecture

### RAG Pipeline

```
Question → DPR Retriever → [Optional: Reranking] → Generator → Answer
                ↓
        Wikipedia Index (21M passages)
```

### Reranking Strategies

1. **Basic**: Cross-encoder scoring only
2. **Enhanced**: 50% semantic + 30% TF-IDF + 20% coverage (recommended)
3. **Diversity**: MMR-style selection for diverse document coverage

### RAG Fusion

1. Generate multiple query variations from original question
2. Retrieve documents for each variation
3. Fuse results using Reciprocal Rank Fusion (RRF)
4. Generate answer from fused context

---

## Hardware Constraints

**Paper settings** (44.5% EM):
- 50 retrieved documents
- 16GB+ V100 GPUs

**Our constraints** (27% EM baseline):
- 15 retrieved documents (8GB GPU limit)
- RTX 2080 Super (8GB VRAM)

All models evaluated fairly with identical 15-doc retrieval.

---

## Troubleshooting

### PyArrow version issue
```
Error: module 'pyarrow' has no attribute 'PyExtensionType'
Fix: pip install pyarrow==13.0.0
```

### Datasets library loading fails
```
Error: NotImplementedError: Loading a dataset cached in a LocalFileSystem
Fix: pip install datasets==2.14.0
```

### CUDA out of memory
```
Solutions:
- Use --n_docs 15 (default)
- Reduce --batch_size to 2 or 1
- Use --use_4bit for large models
```

---

## References

- **Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- **Models**:
  - [facebook/rag-sequence-nq](https://huggingface.co/facebook/rag-sequence-nq)
  - [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
  - [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
- **Dataset**: [Natural Questions](https://huggingface.co/datasets/google-research-datasets/nq_open)

## Hardware

- OS: Windows 11
- CPU: AMD Ryzen 9 3900X (12-core)
- RAM: 48GB
- GPU: RTX 2080 Super (8GB VRAM)
- Python: 3.11.9
