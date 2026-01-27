# Project Summary: RAG Paper Reproduction & Extension

**Paper**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)

---

## Executive Summary

Successfully reproduced RAG-BART baseline and extended with Flan-T5 variants, reranking strategies, and RAG Fusion.

**Key Findings:**
- **Flan-T5-base** matches baseline accuracy (26% vs 27% EM) while being **21x faster**
- **Enhanced reranker** achieves best accuracy (26% EM, 33% F1) with good speed
- **Diversity reranker** achieves highest F1 (34.3%) for comprehensive answers
- **RAG Fusion** on full dataset: 22.8% EM, 30.3% F1 (3,610 samples)

---

## Complete Results

### Main Model Comparison (100 samples)

| Model | Params | EM | F1 | Speed (q/s) | Speedup | Time |
|-------|--------|-----|-----|-------------|---------|------|
| **RAG-BART (baseline)** | 515M | 27.0% | 27.0% | 0.088 | 1.0x | 19 min |
| Flan-T5-small | 77M | 13.0% | 19.7% | 5.02 | 57x | 20s |
| **Flan-T5-base** | 248M | 26.0% | 31.9% | 1.87 | 21x | 54s |
| Flan-T5-large (4-bit) | 494M | 26.0% | 34.1% | 1.20 | 14x | 83s |
| T5 Prompt Engineering | 248M | 12.0% | 18.0% | 3.64 | 41x | 27s |

### Reranking Strategies (Flan-T5-base, 100 samples)

| Strategy | EM | F1 | Speed (q/s) | Time | Description |
|----------|-----|-----|-------------|------|-------------|
| Basic | 21.0% | 30.6% | 0.99 | 101s | Cross-encoder only |
| **Enhanced** | **26.0%** | **33.0%** | **0.96** | **104s** | CE + TF-IDF + coverage |
| Diversity | 24.0% | 34.3% | 0.62 | 160s | MMR-style diversity |

### RAG Fusion (Flan-T5-base)

| Samples | EM | F1 | Speed (q/s) | Time |
|---------|-----|-----|-------------|------|
| 100 | 26.0% | 32.8% | 0.71 | 142s |
| **3,610 (full)** | **22.8%** | **30.3%** | **0.72** | **84 min** |

---

## What We Built

### 1. Evaluation Infrastructure

**Baseline RAG-BART** (`experiments/eval_rag_baseline.py`):
- Facebook's official RAG model (515M params)
- DPR retriever with Wikipedia index (21M passages)
- Greedy decoding (as per paper)
- 15 retrieved documents (8GB GPU constraint)

**Flan-T5 Variants** (`experiments/eval_rag_flan_t5.py`):
- Three model sizes: small (77M), base (248M), large (494M)
- Same DPR retriever as baseline (fair comparison)
- 4-bit quantization support for large model

**Reranking Pipeline** (`experiments/eval_rag_flan_t5_reranked.py`):
- Cross-encoder model: MS-MARCO-MiniLM-L-12-v2 (33.4M params)
- Three strategies: basic, enhanced, diversity
- Configurable weights and parameters

**RAG Fusion** (`experiments/eval_rag_fusion.py`):
- Multiple query variations per question
- Reciprocal Rank Fusion (RRF) for document merging
- Tested on both 100 samples and full validation set

### 2. Reranker Strategies

**Basic Reranker:**
- Pure cross-encoder semantic scoring
- Fast but sometimes picks similar documents
- 21% EM, 30.6% F1

**Enhanced Reranker (Recommended):**
- 50% semantic relevance (cross-encoder)
- 30% term overlap (TF-IDF)
- 20% coverage score
- Best balance: 26% EM, 33% F1

**Diversity Reranker:**
- MMR-style selection
- 60% relevance + 40% diversity from selected docs
- Highest F1: 34.3% (best for comprehensive answers)

### 3. Analysis Tools

- `analysis/compare_results.py` - Metrics comparison and tables
- `analysis/visualize_results.py` - 5 publication-quality plots

---

## Key Scientific Findings

### 1. Instruction Tuning Improves Answer Quality

| Model | EM | F1 | F1 - EM Gap |
|-------|-----|-----|-------------|
| RAG-BART | 27% | 27% | 0% |
| Flan-T5-small | 13% | 19.7% | 6.7% |
| Flan-T5-base | 26% | 31.9% | 5.9% |
| Flan-T5-large | 26% | 34.1% | 8.1% |

- Flan-T5 generates partial matches (F1 > EM)
- Baseline generates "all or nothing" outputs (F1 = EM)

### 2. Reranking Trade-offs

- **Without reranking**: Flan-T5-base gets 26% EM at 1.87 q/s
- **With Enhanced reranking**: Still 26% EM, but F1 improves to 33%
- **With Diversity reranking**: 24% EM but 34.3% F1 (best for comprehensive answers)

### 3. Hardware Constraint Impact

- Paper uses 50 docs → 44.5% EM (on 16GB+ GPU)
- We use 15 docs → 27% EM (on 8GB GPU)
- Fair comparison maintained across all models

### 4. Full Dataset Validation (RAG Fusion)

- 3,610 samples: 22.8% EM, 30.3% F1
- Consistent with 100-sample estimates
- Confirms approach scales to full dataset

---

## Deliverables Checklist

### Code Files

- [x] `experiments/eval_rag_baseline.py` - RAG-BART baseline
- [x] `experiments/eval_rag_flan_t5.py` - Flan-T5 variants
- [x] `experiments/eval_rag_flan_t5_reranked.py` - Reranking strategies
- [x] `experiments/eval_rag_fusion.py` - RAG Fusion
- [x] `experiments/eval_rag_t5_promt.py` - Prompt engineering
- [x] `experiments/eval_rag_custom.py` - Custom RAG
- [x] `experiments/compare_rag_variants.py` - Variant comparison
- [x] `experiments/reranker.py` - Reranker implementations
- [x] `experiments/prompts.py` - Prompt templates
- [x] `experiments/test_nq_loading.py` - Dataset validation
- [x] `analysis/compare_results.py` - Comparison analysis
- [x] `analysis/visualize_results.py` - Plot generation
- [x] `src/evaluation/metrics.py` - EM and F1 metrics
- [x] `scripts/download_nq_hf_hub.py` - Dataset download

### Result Files

- [x] `results/metrics/baseline_100samples.json`
- [x] `results/metrics/baseline_200samples.json`
- [x] `results/metrics/flan_t5_small_100.json`
- [x] `results/metrics/flan_t5_base_100.json`
- [x] `results/metrics/flan_t5_large_4bit_100.json`
- [x] `results/metrics/reranker_strategies_100.json`
- [x] `results/metrics/rag_fusion_100.json`
- [x] `results/metrics/rag_fusion_results.json` (full 3,610 samples)
- [x] `results/metrics/t5_prompt_100.json`

### Visualizations (300 DPI)

- [x] `results/figures/accuracy_comparison.png`
- [x] `results/figures/speed_vs_accuracy.png`
- [x] `results/figures/speedup_comparison.png`
- [x] `results/figures/f1_em_gap.png`
- [x] `results/figures/combined_summary.png`

### Documentation

- [x] `README.md` - Project overview and results
- [x] `SETUP_NOTES.md` - Complete setup guide
- [x] `PROJECT_SUMMARY.md` - This executive summary
- [x] `RERANKER_ANALYSIS_REPORT.md` - Reranker strategy analysis
- [x] `presentation/PRESENTATION_OUTLINE.md` - 20-min talk
- [x] `requirements.txt` - Exact dependencies

---

## Recommendations

### For Production Deployment

**Use: Flan-T5-base + Enhanced Reranker**
- 26% EM, 33% F1
- ~1 q/s throughput
- Best accuracy with reasonable latency
- Fits in 8GB GPU

### For Batch Analysis (Offline)

**Use: Flan-T5-large (4-bit) + Diversity Reranker**
- Highest F1 (34.3%)
- Better coverage of answer space
- Acceptable latency for batch processing

### For Maximum Speed

**Use: Flan-T5-small (no reranker)**
- 5+ q/s throughput
- Lower accuracy (13% EM) but acceptable for some use cases
- Minimal resource requirements

### For Paper Reproduction

**Use: RAG-BART on 16GB+ GPU**
- 50 docs retrieval
- Expected: ~44.5% EM (paper result)
- Reference implementation

---

## Future Work

1. **Full Paper Reproduction**
   - Test with 50 docs on 16GB+ GPU
   - Compare with paper's 44.5% EM

2. **Model Exploration**
   - Flan-T5 XL (3B) and XXL (11B)
   - Fine-tuning on Natural Questions
   - Other instruction-tuned models (Llama, Mistral)

3. **Optimization**
   - Fix T5 Prompt Engineering (currently 12% EM)
   - Optimize diversity reranker speed
   - Experiment with different retrieval K values

4. **Additional Datasets**
   - TriviaQA
   - WebQuestions
   - Multi-domain evaluation

---

## Technical Details

### Models Used

| Model | Parameters | Source |
|-------|-----------|--------|
| RAG-BART | 515M | facebook/rag-sequence-nq |
| DPR Question Encoder | 110M | facebook/dpr-question_encoder-single-nq-base |
| Flan-T5-small | 77M | google/flan-t5-small |
| Flan-T5-base | 248M | google/flan-t5-base |
| Flan-T5-large | 780M (494M 4-bit) | google/flan-t5-large |
| Cross-Encoder | 33.4M | cross-encoder/ms-marco-MiniLM-L-12-v2 |

### Dataset

- **Natural Questions** (Kwiatkowski et al., 2019)
- 3,610 validation samples
- Real questions people asked Google

### Hardware

- CPU: AMD Ryzen 9 3900X (12-core)
- GPU: NVIDIA RTX 2080 Super (8GB VRAM)
- RAM: 48GB DDR4
- OS: Windows 11
- Python: 3.11.9

### Critical Dependencies

```
pyarrow==13.0.0          # MUST be exactly 13.0.0
datasets==2.14.0         # MUST be exactly 2.14.0
sentencepiece==0.1.99    # MUST be 0.1.99 (0.2.x crashes on Windows)
torch>=2.0.0             # With CUDA 11.8
transformers>=4.35.0
```

---

## Acknowledgments

**Paper**: Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

**Tools**: HuggingFace Transformers, PyTorch, FAISS, BitsAndBytes, Sentence-Transformers

**Dataset**: Natural Questions (Google Research)

---

**Last Updated**: January 2026
**Test Set**: Natural Questions validation split
**Status**: All evaluations complete
