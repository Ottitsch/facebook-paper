"""
Baseline RAG evaluation script for Natural Questions.
Based on the official HuggingFace transformers RAG implementation.
"""

import argparse
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers.models.rag.modeling_rag import RagSequenceForGeneration
from transformers.models.rag.tokenization_rag import RagTokenizer
from transformers.models.rag.retrieval_rag import RagRetriever
from datasets import load_dataset
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


def load_rag_model(model_name="facebook/rag-sequence-nq", use_fp16=True, n_docs=15, use_dummy=False):
    """
    Load RAG model with retriever.

    Args:
        model_name: HuggingFace model name
        use_fp16: Use FP16 precision
        n_docs: Number of documents to retrieve
        use_dummy: Use dummy dataset (for testing only)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading RAG model: {model_name}")
    print(f"Using FP16: {use_fp16}, n_docs: {n_docs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RagTokenizer.from_pretrained(model_name)

    # Load retriever first with compressed index
    if use_dummy:
        print("Loading retriever with DUMMY dataset (for testing only - results will be poor)...")
        retriever = RagRetriever.from_pretrained(
            model_name,
            index_name="compressed",
            use_dummy_dataset=True
        )
    else:
        print("Loading retriever with compressed Wikipedia index (this may take several minutes)...")
        print("This will download the Wikipedia index (~36GB compressed)...")
        retriever = RagRetriever.from_pretrained(
            model_name,
            index_name="compressed",
        )

    # Load model with the retriever
    print("Loading RAG model...")
    model = RagSequenceForGeneration.from_pretrained(
        model_name,
        retriever=retriever,
        torch_dtype=torch.float16 if (use_fp16 and device == "cuda") else torch.float32,
    )

    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Configure generation parameters
    model.config.n_docs = n_docs

    print(f"Model loaded successfully on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    return model, tokenizer


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
    import pandas as pd
    from pathlib import Path

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


def evaluate_rag(model, tokenizer, dataset, batch_size=4, max_length=50):
    """
    Evaluate RAG model on Natural Questions.

    Args:
        model: RAG model
        tokenizer: RAG tokenizer
        dataset: Dataset to evaluate on
        batch_size: Batch size for generation
        max_length: Maximum generation length

    Returns:
        Dictionary with results
    """
    device = next(model.parameters()).device
    predictions = []
    references = []

    start_time = time.time()

    print(f"\nEvaluating on {len(dataset)} examples...")

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]

        # Extract questions - handle different dataset formats
        if isinstance(batch, dict):
            # Single example
            questions = [batch.get('question', batch.get('query', ''))]
            answers = [batch.get('answer', batch.get('answers', []))]
        else:
            # Multiple examples
            questions = [ex.get('question', ex.get('query', '')) for ex in batch]
            answers = [ex.get('answer', ex.get('answers', [])) for ex in batch]

        # Prepare inputs
        inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate (using greedy decoding as per paper for QA tasks)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                do_sample=False  # Greedy decoding (paper: "we use greedy decoding for QA")
            )

        # Decode predictions
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)

        # Process answers - ensure they're lists
        for ans in answers:
            if isinstance(ans, str):
                references.append([ans])
            elif isinstance(ans, list):
                references.append(ans if ans else [''])  # Handle empty lists
            else:
                references.append([''])

    # Compute metrics
    end_time = time.time()
    elapsed = end_time - start_time

    metrics = compute_metrics(predictions, references)
    metrics['total_time'] = elapsed
    metrics['questions_per_second'] = len(dataset) / elapsed

    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG on Natural Questions")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/rag-sequence-nq",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
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
        "--n_docs",
        type=int,
        default=15,
        help="Number of documents to retrieve (paper uses 50, but 15 fits in 8GB GPU)"
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
        "--use_dummy_index",
        action="store_true",
        help="Use dummy dataset for testing (poor results but fast)"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_rag_model(
        model_name=args.model_name,
        use_fp16=not args.no_fp16,
        n_docs=args.n_docs,
        use_dummy=args.use_dummy_index
    )

    # Load dataset
    dataset = load_natural_questions(
        split=args.split,
        max_samples=args.max_samples
    )

    # Evaluate
    results = evaluate_rag(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Dataset: Natural Questions ({args.split})")
    print(f"Number of examples: {results['metrics']['num_examples']}")
    print(f"\nExact Match: {results['metrics']['exact_match']:.2f}%")
    print(f"F1 Score: {results['metrics']['f1']:.2f}%")
    print(f"\nTotal time: {results['metrics']['total_time']:.2f}s")
    print(f"Speed: {results['metrics']['questions_per_second']:.2f} questions/second")
    print("="*50)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Don't save predictions/references to keep file size small
        save_results = {
            'model_name': args.model_name,
            'split': args.split,
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
        print(f"  Reference: {results['references'][i][0]}")
        em = 1 if results['predictions'][i].lower().strip() == results['references'][i][0].lower().strip() else 0
        print(f"  Match: {'YES' if em else 'NO'}")


if __name__ == "__main__":
    main()
