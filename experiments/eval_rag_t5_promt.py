"""
EXTENSION: Prompt Engineering Experiment.
Inherits setup from eval_rag_flan_t5.py but overrides the evaluation loop.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# 1. Setup Path to import from sibling files
# We need to add the project root to sys.path to see 'experiments' and 'src'
sys.path.insert(0, str(Path(__file__).parent.parent))

# 2. MODULAR IMPORT: Reuse your existing stable code
# This avoids code duplication for model loading and data processing
try:
    from experiments.eval_rag_flan_t5 import (load_flan_t5_with_retriever,
                                              load_natural_questions)
    from src.evaluation.metrics import compute_metrics
except ImportError:
    print("Error: Could not import from eval_rag_flan_t5. Make sure you are running from project root.")
    sys.exit(1)


# --- NEW EXTENSION LOGIC ---

PROMPT_TEMPLATES = {
    "standard": "{context}\n\nQuestion: {question}\n\nAnswer:",
    
    "reasoning": (
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question above using only the context provided. Provide your answer:"
    ),
    
    "instruction": (
        "{context}\n\n"
        "Q: {question}\n"
        "A:"
    )
}

def format_custom_prompt(question, docs, prompt_type="standard"):
    """
    Custom formatting logic that overrides the baseline simple format.
    """
    # Extract text from docs list
    contexts = [d['text'] for d in docs if 'text' in d]
    context_text = " ".join(contexts)
    
    # Get template
    template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["standard"])
    
    return template.format(question=question, context=context_text)


def evaluate_with_prompts(model, tokenizer, retriever, question_encoder, question_encoder_tokenizer,
                          dataset, batch_size=4, max_length=50, n_docs=15, prompt_type="standard"):
    """
    Modified evaluation loop that uses custom prompts.
    """
    device = next(model.parameters()).device
    predictions = []
    references = []
    
    print(f"\nRunning Extension: '{prompt_type.upper()}' Prompting")
    print(f"Evaluating on {len(dataset)} examples...")
    print(f"Retrieving {n_docs} docs per question...")

    start_time = time.time()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating", total=int(len(dataset)/batch_size)+1):
        try:
            batch = dataset[i:i+batch_size]
            
            # Unpack batch (handles both dict and list format)
            if isinstance(batch, dict):
                questions = [batch.get('question', batch.get('query', ''))]
                answers = [batch.get('answer', batch.get('answers', []))]
            else:
                questions = [ex.get('question', ex.get('query', '')) for ex in batch]
                answers = [ex.get('answer', ex.get('answers', [])) for ex in batch]

            # --- RETRIEVAL STEP (Standard) ---
            batch_prompts = []
            for question in questions:
                # 1. Encode Question
                q_enc = question_encoder_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    q_embed = question_encoder(q_enc.input_ids.to(device), attention_mask=q_enc.attention_mask.to(device)).pooler_output

                # 2. Retrieve Docs
                retrieved = retriever(q_enc.input_ids.to(device), q_embed.cpu().numpy(), n_docs=n_docs, return_tensors="pt")
                doc_ids = retrieved["doc_ids"][0]
                docs_dicts = retriever.index.get_doc_dicts(doc_ids)
                docs = [{"text": doc["text"]} for doc in docs_dicts]

                # 3. FORMAT PROMPT (The Extension Part)
                prompt = format_custom_prompt(question, docs, prompt_type=prompt_type)
                batch_prompts.append(prompt)

            # --- GENERATION STEP (Standard) ---
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=False # Greedy decoding for reproducibility
                )

            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Extract only the answer part (after "Answer:" if present)
            cleaned_preds = []
            for pred in batch_preds:
                if "Answer:" in pred:
                    # Take only the text after the last "Answer:" prompt
                    answer = pred.split("Answer:")[-1].strip()
                else:
                    answer = pred.strip()
                cleaned_preds.append(answer)
            predictions.extend(cleaned_preds)
            
            # Standardize answers for metric computation
            for ans in answers:
                if isinstance(ans, str): references.append([ans])
                elif isinstance(ans, list): references.append(ans if ans else [''])
                else: references.append([''])
        except Exception as e:
            print(f"\nError processing batch at index {i}: {e}")
            continue
        except Exception as e:
            print(f"\nError processing batch at index {i}: {e}")
            print(f"Continuing with {len(predictions)} samples collected so far...")
            continue

    # Compute Metrics
    elapsed = time.time() - start_time
    metrics = compute_metrics(predictions, references)
    metrics['total_time'] = elapsed
    metrics['questions_per_second'] = len(dataset) / elapsed
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="RAG Extension: Prompt Engineering")
    # Reusing the same args structure for consistency
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--prompt_type", type=str, default="standard", choices=PROMPT_TEMPLATES.keys())
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--n_docs", type=int, default=15)
    parser.add_argument("--use_4bit", action="store_true")
    
    args = parser.parse_args()

    # 1. REUSE: Load Model & Data using imported functions
    # We don't write this code again!
    print("\n" + "="*50)
    print("LOADING MODELS & RETRIEVERS")
    print("="*50)
    print("[1/2] Loading Flan-T5 model with DPR retriever...")
    model, tokenizer, retriever, q_enc, q_tok = load_flan_t5_with_retriever(
        model_name=args.model_name,
        n_docs=args.n_docs,
        use_4bit=args.use_4bit
    )
    print("      [OK] Model and retriever loaded")
    
    print("[2/2] Loading Natural Questions dataset...")
    dataset = load_natural_questions(split="validation", max_samples=args.max_samples)
    print(f"      [OK] Dataset loaded ({len(dataset)} samples)")
    print("="*50 + "\n")

    # 2. RUN: Execute the modified evaluation loop
    print("RUNNING EVALUATION")
    print("="*50)
    metrics = evaluate_with_prompts(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        question_encoder=q_enc,
        question_encoder_tokenizer=q_tok,
        dataset=dataset,
        prompt_type=args.prompt_type,
        n_docs=args.n_docs
    )

    # 3. REPORT
    print("="*50)
    print(f"EXTENSION RESULTS: {args.prompt_type.upper()}")
    print("="*50)
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score:    {metrics['f1']:.2f}%")
    print("="*50)
    print("[OK] Evaluation completed successfully\n")

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump({'args': vars(args), 'metrics': metrics}, f, indent=2)
        print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()