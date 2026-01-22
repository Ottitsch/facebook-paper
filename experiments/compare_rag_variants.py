"""
Compare different RAG configurations and prompt strategies.
Systematically evaluates combinations of prompt types and reranking strategies.
Outputs CSV comparison and console summary statistics.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments.eval_rag_flan_t5 import (load_flan_t5_with_retriever,
                                          load_natural_questions)
from experiments.eval_rag_t5_promt import evaluate_with_prompts
from experiments.prompts import get_available_prompts
from experiments.reranker import load_reranker, rerank_documents
from src.evaluation.metrics import compute_metrics


def evaluate_with_prompts_and_reranking(model, tokenizer, retriever, question_encoder, 
                                        question_encoder_tokenizer, reranker, dataset, 
                                        batch_size=4, max_length=50, prompt_type="standard",
                                        use_reranker=True, retrieval_k=50, rerank_top_k=15,
                                        reranker_strategy="enhanced"):
    """
    Evaluate model with custom prompts and optional reranking with configurable strategy.
    
    Args:
        model: Flan-T5 model
        tokenizer: T5 tokenizer
        retriever: DPR retriever
        question_encoder: DPR question encoder
        question_encoder_tokenizer: DPR tokenizer
        reranker: Reranker instance
        dataset: Natural Questions dataset
        batch_size: Batch size for generation
        max_length: Max generation length
        prompt_type: Type of prompt to use
        use_reranker: Whether to use reranking
        retrieval_k: Number of docs to retrieve
        rerank_top_k: Number of docs to keep after reranking
        reranker_strategy: Reranker strategy ("basic", "enhanced", "diversity")
    
    Returns:
        Dict with metrics and predictions
    """
    device = next(model.parameters()).device
    predictions = []
    references = []
    retrieval_times = []
    reranking_times = []
    
    # Load reranker with specified strategy if reranking is enabled
    if use_reranker:
        reranker = load_reranker(strategy=reranker_strategy, use_fp16=True)
    
    start_time = time.time()
    n_docs = retrieval_k if use_reranker else rerank_top_k
    
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Eval (Prompt={prompt_type}, Rerank={reranker_strategy if use_reranker else 'None'})"):
        batch = dataset[i:i+batch_size]
        
        # Unpack batch
        if isinstance(batch, dict):
            questions = [batch.get('question', batch.get('query', ''))]
            answers = [batch.get('answer', batch.get('answers', []))]
        else:
            questions = [ex.get('question', ex.get('query', '')) for ex in batch]
            answers = [ex.get('answer', ex.get('answers', [])) for ex in batch]

        for question, answer_list in zip(questions, answers):
            # Retrieval
            ret_start = time.time()
            q_enc = question_encoder_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                q_embed = question_encoder(q_enc.input_ids.to(device), 
                                          attention_mask=q_enc.attention_mask.to(device)).pooler_output

            retrieved = retriever(q_enc.input_ids.to(device), q_embed.cpu().numpy(), 
                                 n_docs=n_docs, return_tensors="pt")
            doc_ids = retrieved["doc_ids"][0]
            
            # Convert to document dicts using retriever index
            docs_list = retriever.index.get_doc_dicts(doc_ids)
            docs = [{"id": doc_id.item(), "text": docs_list[j]["text"]} 
                   for j, doc_id in enumerate(doc_ids)]
            
            ret_time = time.time() - ret_start
            retrieval_times.append(ret_time * 1000)  # ms
            
            # Reranking (optional)
            if use_reranker and docs:
                rerank_start = time.time()
                docs = reranker.rerank(question, docs, top_k=rerank_top_k, batch_size=32)
                rerank_time = time.time() - rerank_start
                reranking_times.append(rerank_time * 1000)  # ms
            else:
                reranking_times.append(0)
            
            # Format prompt with retrieved (and optionally reranked) docs
            from experiments.eval_rag_t5_promt import format_custom_prompt
            prompt = format_custom_prompt(question, docs[:rerank_top_k], prompt_type=prompt_type)
            
            # Generation
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids.to(device),
                    max_length=max_length,
                    num_beams=1,
                    early_stopping=True
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
            references.append(answer_list if isinstance(answer_list, list) else [answer_list])

    total_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(predictions, references)
    metrics['total_time'] = total_time
    metrics['avg_retrieval_time_ms'] = np.mean(retrieval_times)
    metrics['avg_reranking_time_ms'] = np.mean(reranking_times)
    metrics['questions_per_second'] = len(predictions) / total_time if total_time > 0 else 0
    
    return {
        'predictions': predictions,
        'references': references,
        'metrics': metrics
    }


def compare_rag_variants(model_name="google/flan-t5-base", dataset_split="validation",
                         max_samples=100, batch_size=4, output_dir="results/rag_variants",
                         reranker_strategies=None):
    """
    Compare all combinations of prompt types and reranking strategies.

    Evaluates each combination of:
    - Prompt types (standard, reasoning, instruction, etc.)
    - Reranking disabled vs enabled with specified strategies

    Args:
        model_name: HuggingFace model name
        dataset_split: Train or validation split
        max_samples: Number of samples to evaluate
        batch_size: Batch size for generation
        output_dir: Directory for output files
        reranker_strategies: List of reranker strategies to compare
                            (default: ["enhanced", "diversity"])
                            Options: "basic", "enhanced", "diversity"

    Returns:
        DataFrame with comparison results
    """
    if reranker_strategies is None:
        reranker_strategies = ["enhanced", "diversity"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load models and data once
    print("="*50)
    print("LOADING MODELS AND DATA")
    print("="*50)

    model, tokenizer, retriever, question_encoder, question_encoder_tokenizer = load_flan_t5_with_retriever(
        model_name=model_name,
        use_fp16=True,
        n_docs=50,
        use_4bit=False
    )

    reranker = load_reranker(use_fp16=True)

    dataset = load_natural_questions(split=dataset_split, max_samples=max_samples)

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_split} ({len(dataset)} samples)")
    print(f"Reranker strategies: {', '.join(reranker_strategies)}")
    print()

    # Define configurations
    prompt_types = get_available_prompts()
    reranker_configs = [
        {"use_reranker": False, "retrieval_k": 15, "rerank_top_k": 15, "reranker_strategy": None},
    ]
    
    # Add configurations for each specified reranker strategy
    for strategy in reranker_strategies:
        reranker_configs.append({
            "use_reranker": True,
            "retrieval_k": 50,
            "rerank_top_k": 15,
            "reranker_strategy": strategy
        })

    results = []
    total_configs = len(prompt_types) * len(reranker_configs)
    current = 0

    print("="*50)
    print("RUNNING EVALUATIONS")
    print("="*50)
    print()

    for prompt_type in prompt_types:
        for reranker_config in reranker_configs:
            current += 1

            strategy_str = reranker_config['reranker_strategy'] if reranker_config['reranker_strategy'] else "None"
            print(f"[{current}/{total_configs}] Prompt={prompt_type}, Reranker_Strategy={strategy_str}")

            try:
                result = evaluate_with_prompts_and_reranking(
                    model=model,
                    tokenizer=tokenizer,
                    retriever=retriever,
                    question_encoder=question_encoder,
                    question_encoder_tokenizer=question_encoder_tokenizer,
                    reranker=reranker,
                    dataset=dataset,
                    batch_size=batch_size,
                    max_length=50,
                    prompt_type=prompt_type,
                    **reranker_config
                )

                # Extract key metrics
                metrics = result['metrics']

                row = {
                    'prompt_type': prompt_type,
                    'use_reranker': reranker_config['use_reranker'],
                    'reranker_strategy': reranker_config.get('reranker_strategy', 'None'),
                    'retrieval_k': reranker_config['retrieval_k'],
                    'rerank_top_k': reranker_config['rerank_top_k'],
                    'exact_match': metrics['exact_match'],
                    'f1_score': metrics['f1'],
                    'num_examples': metrics['num_examples'],
                    'questions_per_second': metrics['questions_per_second'],
                    'total_time': metrics['total_time'],
                    'avg_retrieval_time_ms': metrics['avg_retrieval_time_ms'],
                    'avg_reranking_time_ms': metrics['avg_reranking_time_ms'],
                }

                results.append(row)

                print(f"  EM: {metrics['exact_match']:.2f}% | F1: {metrics['f1']:.2f}% | Speed: {metrics['questions_per_second']:.2f} q/s")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue

            print()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = output_path / "rag_variants_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)

    print("\nBest by Exact Match:")
    best_em = df.loc[df['exact_match'].idxmax()]
    print(f"  {best_em['prompt_type']} (Rerank: {best_em['use_reranker']}) → {best_em['exact_match']:.2f}% EM")

    print("\nBest by Speed:")
    best_speed = df.loc[df['questions_per_second'].idxmax()]
    print(f"  {best_speed['prompt_type']} (Rerank: {best_speed['use_reranker']}) → {best_speed['questions_per_second']:.2f} q/s")

    print("\nBest EM/Speed Balance:")
    df['balance_score'] = (df['exact_match'] / df['exact_match'].max()) * (df['questions_per_second'] / df['questions_per_second'].max())
    best_balance = df.loc[df['balance_score'].idxmax()]
    print(f"  {best_balance['prompt_type']} (Rerank: {best_balance['use_reranker']}) → EM={best_balance['exact_match']:.2f}%, Speed={best_balance['questions_per_second']:.2f} q/s")

    print("\n" + "="*50)

    return df


def main():
    parser = argparse.ArgumentParser(description="Compare RAG variants (prompts + reranking)")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rag_variants",
        help="Output directory for results"
    )
    parser.add_argument(
        "--reranker_strategies",
        type=str,
        nargs='+',
        default=["enhanced", "diversity"],
        choices=["basic", "enhanced", "diversity"],
        help="Reranker strategies to compare (space-separated)"
    )

    args = parser.parse_args()

    df = compare_rag_variants(
        model_name=args.model_name,
        dataset_split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        reranker_strategies=args.reranker_strategies
    )


if __name__ == "__main__":
    main()
