"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-Flan-T5 RAG evaluation with reranking on Natural Questions.
Uses DPR retriever + cross-encoder reranker with Flan-T5 generator.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from experiments.eval_rag_flan_t5 import (format_t5_prompt,
                                          load_flan_t5_with_retriever,
                                          load_natural_questions)
from experiments.reranker import load_reranker, rerank_documents
from src.evaluation.metrics import compute_metrics


def evaluate_flan_t5_reranked(model, tokenizer, retriever, question_encoder, 
                               question_encoder_tokenizer, reranker, dataset, 
                               batch_size=4, max_length=50, retrieval_k=50, rerank_top_k=15):
    """
    Evaluate Flan-T5 with reranking on Natural Questions.
    
    Args:
        model: Flan-T5 model
        tokenizer: T5 tokenizer
        retriever: DPR retriever
        question_encoder: DPR question encoder
        question_encoder_tokenizer: DPR tokenizer
        reranker: Cross-encoder reranker
        dataset: Natural Questions dataset
        batch_size: Batch size for generation
        max_length: Max generation length
        retrieval_k: Number of docs to retrieve
        rerank_top_k: Number of docs to keep after reranking
    
    Returns:
        Dict with metrics and predictions
    """
    device = next(model.parameters()).device
    predictions = []
    references = []
    retrieval_times = []
    reranking_times = []
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        questions = [ex["question"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        
        batch_prompts = []
        
        for question in questions:
            retr_start = time.time()
            
            # Encode question
            q_enc = question_encoder_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                q_embed = question_encoder(
                    q_enc.input_ids.to(device),
                    attention_mask=q_enc.attention_mask.to(device)
                ).pooler_output
            
            # Retrieve more documents
            retrieved = retriever(
                q_enc.input_ids.to(device),
                q_embed.cpu().numpy(),
                n_docs=retrieval_k,
                return_tensors="pt"
            )
            doc_ids = retrieved["doc_ids"][0]
            docs_dicts = retriever.index.get_doc_dicts(doc_ids)
            docs = [{"text": doc["text"]} for doc in docs_dicts]
            
            retrieval_times.append(time.time() - retr_start)
            
            # Reranking step
            rerank_start = time.time()
            reranked_docs = rerank_documents(question, docs, reranker, top_k=rerank_top_k)
            reranking_times.append(time.time() - rerank_start)
            
            # Format prompt
            context_parts = []
            for doc in reranked_docs:
                text = doc.get('text', '')
                if text:
                    context_parts.append(text)
            
            context = " ".join(context_parts)
            
            # T5 format
            prompt = f"question: {question} context: {context}"
            batch_prompts.append(prompt)
        
        # Generation step
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False
            )
        
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)
        
        # Process answers - ensure they're lists
        for ans in answers:
            if isinstance(ans, np.ndarray):
                references.append(ans.tolist() if len(ans) > 0 else [''])
            elif isinstance(ans, str):
                references.append([ans])
            elif isinstance(ans, list):
                references.append(ans if ans else [''])
            else:
                references.append([''])
    
    # Compute metrics
    end_time = time.time()
    elapsed = end_time - start_time
    
    metrics = compute_metrics(predictions, references)
    metrics['total_time'] = elapsed
    metrics['questions_per_second'] = len(dataset) / elapsed
    metrics['avg_retrieval_time_ms'] = (sum(retrieval_times) / len(retrieval_times) * 1000) if retrieval_times else 0
    metrics['avg_reranking_time_ms'] = (sum(reranking_times) / len(reranking_times) * 1000) if reranking_times else 0
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flan-T5 with reranking on Natural Questions")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace Flan-T5 model name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--retrieval_k",
        type=int,
        default=50,
        help="Number of documents to retrieve initially"
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=15,
        help="Number of documents to keep after reranking"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable FP16 precision"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (for flan-t5-large)"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer, retriever, question_encoder, question_encoder_tokenizer = load_flan_t5_with_retriever(
        model_name=args.model_name,
        use_fp16=not args.no_fp16,
        n_docs=args.rerank_top_k,
        use_4bit=args.use_4bit
    )

    # Load reranker
    reranker = load_reranker(use_fp16=not args.no_fp16)

    # Load dataset
    dataset = load_natural_questions(
        split=args.split,
        max_samples=args.max_samples
    )

    # Evaluate
    results = evaluate_flan_t5_reranked(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        question_encoder=question_encoder,
        question_encoder_tokenizer=question_encoder_tokenizer,
        reranker=reranker,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        retrieval_k=args.retrieval_k,
        rerank_top_k=args.rerank_top_k
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Retrieval: {args.retrieval_k} -> Rerank: {args.rerank_top_k}")
    print(f"Number of examples: {results['metrics']['num_examples']}")
    print(f"\nExact Match: {results['metrics']['exact_match']:.2f}%")
    print(f"F1 Score: {results['metrics']['f1']:.2f}%")
    print(f"\nAvg retrieval time: {results['metrics']['avg_retrieval_time_ms']:.1f}ms")
    print(f"Avg reranking time: {results['metrics']['avg_reranking_time_ms']:.1f}ms")
    print(f"Total time: {results['metrics']['total_time']:.2f}s")
    print(f"Speed: {results['metrics']['questions_per_second']:.2f} questions/second")
    print("="*50)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_results = {
            'model_name': args.model_name,
            'retrieval_k': args.retrieval_k,
            'rerank_top_k': args.rerank_top_k,
            'metrics': results['metrics'],
            'args': vars(args)
        }

        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Show some examples
    print("\nExample predictions:")
    for i in range(min(5, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"  Prediction: {results['predictions'][i]}")
        ref_text = results['references'][i][0] if results['references'][i] else ''
        print(f"  Reference: {ref_text}")
        em = 1 if results['predictions'][i].lower().strip() == ref_text.lower().strip() else 0
        print(f"  Match: {'YES' if em else 'NO'}")


if __name__ == "__main__":
    main()
