#!/usr/bin/env python
"""
Standalone reranker strategy comparison.
Runs Enhanced vs Diversity strategies and appends results to comparison_all.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from experiments.eval_rag_flan_t5 import (load_flan_t5_with_retriever,
                                          load_natural_questions)
from experiments.eval_rag_flan_t5_reranked import evaluate_flan_t5_reranked
from experiments.reranker import load_reranker


def run_reranker_enhanced(max_samples: int = 100):
    """Run enhanced reranker strategy."""
    print("\n" + "="*70)
    print("Testing Strategy: ENHANCED")
    print("="*70 + "\n")
    
    model, tokenizer, retriever, question_encoder, question_encoder_tokenizer = load_flan_t5_with_retriever(
        model_name="google/flan-t5-base",
        use_fp16=True,
        n_docs=50,
        use_4bit=False
    )
    
    dataset = load_natural_questions(split="validation", max_samples=max_samples)
    
    reranker = load_reranker(strategy="enhanced", use_fp16=True)
    
    result = evaluate_flan_t5_reranked(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        question_encoder=question_encoder,
        question_encoder_tokenizer=question_encoder_tokenizer,
        reranker=reranker,
        dataset=dataset,
        batch_size=4,
        max_length=50,
        retrieval_k=50,
        rerank_top_k=15
    )
    
    metrics = result['metrics']
    
    print(f"\nENHANCED Results:")
    print(f"  EM: {metrics['exact_match']:.2f}%")
    print(f"  F1: {metrics['f1']:.2f}%")
    print(f"  Speed: {metrics['questions_per_second']:.2f} q/s")
    print(f"  Total Time: {metrics['total_time']:.1f}s")
    
    return metrics


def run_reranker_diversity(max_samples: int = 100):
    """Run diversity reranker strategy."""
    print("\n" + "="*70)
    print("Testing Strategy: DIVERSITY")
    print("="*70 + "\n")
    
    model, tokenizer, retriever, question_encoder, question_encoder_tokenizer = load_flan_t5_with_retriever(
        model_name="google/flan-t5-base",
        use_fp16=True,
        n_docs=50,
        use_4bit=False
    )
    
    dataset = load_natural_questions(split="validation", max_samples=max_samples)
    
    reranker = load_reranker(strategy="diversity", use_fp16=True)
    
    result = evaluate_flan_t5_reranked(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        question_encoder=question_encoder,
        question_encoder_tokenizer=question_encoder_tokenizer,
        reranker=reranker,
        dataset=dataset,
        batch_size=4,
        max_length=50,
        retrieval_k=50,
        rerank_top_k=15
    )
    
    metrics = result['metrics']
    
    print(f"\nDIVERSITY Results:")
    print(f"  EM: {metrics['exact_match']:.2f}%")
    print(f"  F1: {metrics['f1']:.2f}%")
    print(f"  Speed: {metrics['questions_per_second']:.2f} q/s")
    print(f"  Total Time: {metrics['total_time']:.1f}s")
    
    return metrics


def main():
    """Run both reranker strategies and append to results CSV."""
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print("\n" + "█"*70)
    print("█ RERANKER STRATEGY COMPARISON: Enhanced vs Diversity")
    print("█"*70)
    
    # Run both strategies
    metrics_enhanced = run_reranker_enhanced(max_samples)
    metrics_diversity = run_reranker_diversity(max_samples)
    
    # Prepare rows for comparison_all.csv
    comparison_rows = [
        {
            'Model': 'Flan-T5-base + Reranker (Enhanced)',
            'EM': f"{metrics_enhanced['exact_match']:.1f}%",
            'F1': f"{metrics_enhanced['f1']:.1f}%",
            'Speed (q/s)': f"{metrics_enhanced['questions_per_second']:.2f}",
            'Total Time': f"{metrics_enhanced['total_time']:.1f}s"
        },
        {
            'Model': 'Flan-T5-base + Reranker (Diversity)',
            'EM': f"{metrics_diversity['exact_match']:.1f}%",
            'F1': f"{metrics_diversity['f1']:.1f}%",
            'Speed (q/s)': f"{metrics_diversity['questions_per_second']:.2f}",
            'Total Time': f"{metrics_diversity['total_time']:.1f}s"
        }
    ]
    
    # Load existing comparison_all.csv
    comparison_path = Path("results/comparison_all.csv")
    df_existing = pd.read_csv(comparison_path)
    
    # Create new rows dataframe
    df_new_rows = pd.DataFrame(comparison_rows)
    
    # Append to existing
    df_combined = pd.concat([df_existing, df_new_rows], ignore_index=True)
    
    # Save back
    df_combined.to_csv(comparison_path, index=False)
    
    print("\n" + "="*70)
    print("UPDATED comparison_all.csv")
    print("="*70)
    print(df_combined.to_string())
    print(f"\n✓ Results appended to: {comparison_path}\n")


if __name__ == "__main__":
    main()
