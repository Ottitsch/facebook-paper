"""
RAG Fusion Evaluation Script.
1. Generates multiple query variations using Flan-T5.
2. Retrieves documents for ALL variations.
3. Fuses results using Reciprocal Rank Fusion (RRF).
4. Generates final answer.
"""

import argparse
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers.models.rag.retrieval_rag import RagRetriever
from tqdm import tqdm
import pandas as pd

from src.evaluation.metrics import compute_metrics


def load_fusion_models(model_name="google/flan-t5-base", use_fp16=True, use_dummy=False, use_4bit=False):
    """
    Load models for RAG Fusion.
    Fixes AssertionError by using 'exact' index for dummy mode.
    """
    print(f"Loading Models for RAG Fusion...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Generator (Flan-T5)
    print("Loading T5 tokenizer & model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        dtype = torch.float32
        if use_fp16 and device == "cuda":
            dtype = torch.float16

        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        model = model.to(device)

    model.eval()

    # 2. Load DPR Components
    print("Loading DPR question encoder...")
    question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    question_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    question_encoder = question_encoder.to(device)
    question_encoder.eval()

    # 3. Load Retriever
    print("Loading Retriever (Index)...")
    if use_dummy:
        print("Using DUMMY index (Random data)...")
        #Must use 'exact' for dummy, 'compressed' crashes it
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq",
            index_name="exact",
            use_dummy_dataset=True
        )
    else:
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq",
            index_name="compressed",
        )

    return model, tokenizer, retriever, question_encoder, question_encoder_tokenizer

def load_natural_questions(split="validation", max_samples=None):
    """
    Load Natural Questions dataset from local Parquet files.

    Bypasses datasets library glob pattern bug by loading directly from Parquet.

    Args:
        split: Dataset split ('train', 'validation')
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    print(f"Loading Natural Questions dataset (split={split})...")

    # Path to downloaded Parquet files
    base_path = Path(__file__).parent.parent / "data" / "datasets--google-research-datasets--nq_open"

    # Find the snapshot directory (contains the actual parquet files)
    snapshot_dirs = list(base_path.glob("snapshots/*"))

    if not snapshot_dirs:
        raise FileNotFoundError(
            f"NQ-Open dataset not found at {base_path}. "
            f"Please run: python scripts/download_nq_hf_hub.py"
        )

    # Use the first (should be only) snapshot
    snapshot_dir = snapshot_dirs[0]

    # Determine which file to load
    if split == "validation":
        parquet_file = snapshot_dir / "nq_open" / "validation-00000-of-00001.parquet"
    elif split == "train":
        parquet_file = snapshot_dir / "nq_open" / "train-00000-of-00001.parquet"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'validation'")

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"NQ-Open parquet file not found: {parquet_file}. "
            f"Please run: python scripts/download_nq_hf_hub.py"
        )

    print(f"Loading from: {parquet_file}")

    # Load parquet file
    df = pd.read_parquet(parquet_file)

    # Limit samples if requested
    if max_samples is not None:
        df = df.head(max_samples)

    print(f"Loaded {len(df)} examples")

    # Convert to list of dicts (compatible with existing evaluation code)
    dataset = df.to_dict('records')

    # Print sample
    if len(dataset) > 0:
        print(f"Sample question: {dataset[0]['question'][:100]}...")
        print(f"Sample answer: {dataset[0]['answer']}")

    return dataset


def generate_search_queries(model, tokenizer, question, n_variations=3):
    """
    Ask Flan-T5 to generate search query variations.
    """
    device = next(model.parameters()).device

    # Prompt engineering for query generation
    prompt = (
        f"Generate {n_variations} different search queries to answer this question. "
        f"Separate them with newlines.\nQuestion: {question}"
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            do_sample=True,  # Add some creativity
            temperature=0.7
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse the output into a list
    # Flan-T5 usually separates with commas or newlines depending on prompt
    queries = [q.strip() for q in generated_text.split('\n') if len(q.strip()) > 5]

    # Fallback: if splitting failed, just use the generated text as one query
    if not queries:
        queries = [generated_text]

    # Ensure we include the original question too!
    queries.append(question)

    # Deduplicate and limit
    return list(set(queries))[:n_variations + 1]


def reciprocal_rank_fusion(results_dict, k=60):
    """
    Fuse multiple retrieval lists using RRF.

    Args:
        results_dict: {query_string: [doc_dict1, doc_dict2, ...]}
        k: RRF constant (usually 60)

    Returns:
        List of fused document dicts sorted by score
    """
    fused_scores = defaultdict(float)
    doc_map = {}  # Keep track of full doc content by title/id

    for query, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            # Create a unique key (using title + text hash or just text)
            # text as key for simplicity in this dataset
            doc_key = doc['text'][:100]
            doc_map[doc_key] = doc

            # RRF Formula
            fused_scores[doc_key] += 1 / (k + rank)

    # Sort by score descending
    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)

    return [doc_map[key] for key in sorted_keys]


def evaluate_rag_fusion(model, tokenizer, retriever, question_encoder, question_encoder_tokenizer,
                        dataset, batch_size=1, max_length=50, n_docs_per_query=5):
    """
    Evaluate with RAG Fusion.
    """
    device = next(model.parameters()).device
    predictions = []
    references = []

    start_time = time.time()

    print(f"\nEvaluating on {len(dataset)} examples with RAG Fusion...")

    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):

        # Extract question/answer (handle different formats)
        question = example.get('question', example.get('query', ''))
        answers = example.get('answer', example.get('answers', []))

        # 1. GENERATE QUERIES (The "Fusion" setup)
        generated_queries = generate_search_queries(model, tokenizer, question)

        # 2. RETRIEVE FOR EACH QUERY
        all_retrieved_results = {}

        for q in generated_queries:
            # Encode query
            q_inputs = question_encoder_tokenizer(q, return_tensors="pt", truncation=True)
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}

            with torch.no_grad():
                q_hidden = question_encoder(**q_inputs).pooler_output

            # Retrieve
            retrieved = retriever(
                q_inputs['input_ids'],
                q_hidden.cpu().numpy(),
                n_docs=n_docs_per_query,
                return_tensors="pt"
            )

            doc_ids = retrieved["doc_ids"][0]
            docs_dicts = retriever.index.get_doc_dicts(doc_ids)
            all_retrieved_results[q] = [{"text": doc["text"], "title": doc["title"]} for doc in docs_dicts]

        # 3. FUSE RESULTS
        fused_docs = reciprocal_rank_fusion(all_retrieved_results)

        # Take top K fused docs (e.g., top 5)
        final_context_docs = fused_docs[:5]

        # 4. GENERATE ANSWER
        contexts = [d['text'] for d in final_context_docs]
        context_str = " ".join(contexts)
        prompt = f"question: {question} context: {context_str}"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=False
            )

        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(pred)

        # Handle references
        if isinstance(answers, str):
            references.append([answers])
        elif isinstance(answers, list):
            references.append(answers if answers else [''])
        else:
            references.append(answers.tolist() if hasattr(answers, 'tolist') else [''])

    # Metrics
    end_time = time.time()
    elapsed = end_time - start_time

    metrics = compute_metrics(predictions, references)
    metrics['total_time'] = elapsed
    metrics['questions_per_second'] = len(dataset) / elapsed if elapsed > 0 else 0

    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Fusion on Natural Questions")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep 1 for fusion logic)")

    # Fusion specific arg
    parser.add_argument("--n_docs_per_query", type=int, default=5, help="Docs per query variation")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--use_dummy_index", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")

    args = parser.parse_args()

    # 1. Load Models
    model, tokenizer, retriever, q_enc, q_tok = load_fusion_models(
        model_name=args.model_name,
        use_fp16=not args.no_fp16,
        use_dummy=args.use_dummy_index,
        use_4bit=args.use_4bit
    )

    # 2. Load Dataset
    dataset = load_natural_questions(
        split=args.split,
        max_samples=args.max_samples
    )

    # 3. Evaluate
    results = evaluate_rag_fusion(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        question_encoder=q_enc,
        question_encoder_tokenizer=q_tok,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        n_docs_per_query=args.n_docs_per_query
    )

    # 4. Print Results
    print("\n" + "=" * 50)
    print("FUSION EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Dataset: Natural Questions ({args.split})")
    print(f"Number of examples: {results['metrics']['num_examples']}")
    print(f"\nExact Match: {results['metrics']['exact_match']:.2f}%")
    print(f"F1 Score: {results['metrics']['f1']:.2f}%")
    print(f"\nTotal time: {results['metrics']['total_time']:.2f}s")
    print(f"Speed: {results['metrics']['questions_per_second']:.2f} questions/second")
    print("=" * 50)

    # 5. Save Results (JSON)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_results = {
            'model_name': args.model_name,
            'type': 'rag_fusion',
            'split': args.split,
            'metrics': results['metrics'],
            'args': vars(args)
        }

        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # 6. Show Examples
    print("\nExample predictions:")
    for i in range(min(5, len(results['predictions']))):
        print(f"\nExample {i + 1}:")
        print(f"  Prediction: {results['predictions'][i]}")

        # Handle reference printing
        ref = results['references'][i]
        ref_text = ref[0] if len(ref) > 0 else ""
        print(f"  Reference: {ref_text}")

        # Simple EM check for display
        em = 1 if results['predictions'][i].lower().strip() == ref_text.lower().strip() else 0
        print(f"  Match: {'YES' if em else 'NO'}")

if __name__ == "__main__":
    main()