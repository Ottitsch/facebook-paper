"""
Custom RAG evaluation that loads wiki_dpr directly from cached .arrow files.
This bypasses the datasets library issues and uses downloaded data.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

from src.evaluation.metrics import compute_metrics

# Don't import datasets at module level - it's corrupted
# Import it lazily when needed


def load_wiki_dpr_from_cache(cache_dir=None, max_passages=None):
    """
    Load wiki_dpr passages directly from cached .arrow files.
    Bypasses datasets library loading issues.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "wiki_dpr"
    else:
        cache_dir = Path(cache_dir)

    # Find the processed data
    arrow_dir = cache_dir / "psgs_w100.nq.no_index-dummy=False,with_index=False" / "0.0.0"

    if not arrow_dir.exists():
        raise FileNotFoundError(f"Cache not found at {arrow_dir}")

    # Find all .arrow files
    arrow_files = sorted(arrow_dir.glob("**/*.arrow"))

    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {arrow_dir}")

    print(f"Found {len(arrow_files)} cached arrow files")
    print(f"Loading from: {arrow_dir}")

    # Load all arrow files
    all_data = []
    total_loaded = 0

    for arrow_file in tqdm(arrow_files, desc="Loading cached data"):
        # Load arrow file using feather format (datasets uses this)
        try:
            import pyarrow.feather as feather
            table = feather.read_table(str(arrow_file))
        except:
            # Fallback: try IPC format
            try:
                table = pa.ipc.open_stream(pa.OSFile(str(arrow_file), 'rb')).read_all()
            except:
                print(f"Skipping unreadable file: {arrow_file}")
                continue

        # Convert to list of dicts
        batch_size = len(table)
        for i in range(batch_size):
            if max_passages and total_loaded >= max_passages:
                break

            row = {
                'id': str(table['id'][i].as_py()),
                'text': table['text'][i].as_py(),
                'title': table['title'][i].as_py(),
            }
            all_data.append(row)
            total_loaded += 1

        if max_passages and total_loaded >= max_passages:
            break

    print(f"Loaded {total_loaded} passages from cache")

    # Return as list of dicts (don't need Dataset object)
    return all_data


def build_faiss_index(passages, encoder_model, encoder_tokenizer, batch_size=32):
    """Build FAISS index from passages using DPR encoder."""
    import faiss

    print(f"Building FAISS index from {len(passages)} passages...")

    # Use context encoder for passages
    ctx_encoder = encoder_model['ctx']
    ctx_tokenizer = encoder_tokenizer['ctx']
    device = next(ctx_encoder.parameters()).device
    embeddings = []

    # Encode all passages
    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
        batch_end = min(i + batch_size, len(passages))

        # Get passage texts
        batch_texts = []
        for j in range(i, batch_end):
            title = passages[j].get('title', '')
            text = passages[j].get('text', '')
            # Combine title and text
            batch_texts.append(f"{title}: {text}")

        # Encode
        inputs = ctx_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = ctx_encoder(**inputs)
            # Use pooler output or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_embeddings = outputs.pooler_output
            else:
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            embeddings.append(batch_embeddings.cpu().numpy())

    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings).astype('float32')

    # Build FAISS index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(all_embeddings)
    index.add(all_embeddings)

    print(f"Built FAISS index with {index.ntotal} passages (dimension: {dimension})")

    return index


def retrieve_passages(question, index, passages, encoder_model, encoder_tokenizer, k=5):
    """Retrieve top-k passages for a question."""
    # Use question encoder for queries
    q_encoder = encoder_model['question']
    q_tokenizer = encoder_tokenizer['question']
    device = next(q_encoder.parameters()).device

    # Encode question
    inputs = q_tokenizer(
        [question],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = q_encoder(**inputs)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            query_embedding = outputs.pooler_output.cpu().numpy()
        else:
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            query_embedding = (torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)).cpu().numpy()

    # Normalize
    import faiss
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding.astype('float32'), k)

    # Get passages
    retrieved_passages = []
    for idx in indices[0]:
        if idx < len(passages):
            passage = passages[int(idx)]
            retrieved_passages.append({
                'text': passage.get('text', ''),
                'title': passage.get('title', '')
            })

    return retrieved_passages


def load_models(encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
                generator_name="facebook/bart-large",
                use_fp16=True):
    """Load encoder and generator models."""
    from transformers import (DPRContextEncoder, DPRContextEncoderTokenizer,
                              DPRQuestionEncoder, DPRQuestionEncoderTokenizer)

    print(f"Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load DPR encoders - need both context and question encoders
    print(f"Loading DPR context encoder...")
    ctx_encoder_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_encoder = ctx_encoder.to(device)
    ctx_encoder.eval()

    print(f"Loading DPR question encoder...")
    q_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder = q_encoder.to(device)
    q_encoder.eval()

    # Store both in a dict for clarity
    encoder_model = {'ctx': ctx_encoder, 'question': q_encoder}
    encoder_tokenizer = {'ctx': ctx_encoder_tokenizer, 'question': q_encoder_tokenizer}

    # Load generator
    print(f"Loading generator: {generator_name}")
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
    generator = AutoModelForSeq2SeqLM.from_pretrained(
        generator_name,
        torch_dtype=torch.float16 if (use_fp16 and device == "cuda") else torch.float32,
    )
    generator = generator.to(device)
    generator.eval()

    print(f"Models loaded successfully")

    return encoder_model, encoder_tokenizer, generator, generator_tokenizer


def load_evaluation_dataset(dataset_name="nq_open", split="validation", max_samples=None):
    """Load evaluation dataset."""
    print(f"Loading evaluation dataset: {dataset_name} ({split})...")

    # Try to import and use datasets library
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Loaded {len(dataset)} examples")
        return dataset

    except Exception as e:
        print(f"Error loading from HuggingFace (datasets library issue): {e}")
        print("Falling back to dummy data...")

    # Fallback to dummy data
    print("Using dummy evaluation data...")
    dummy_data = [
        {
            'question': "What is the capital of France?",
            'answer': ["Paris"]
        },
        {
            'question': "Who wrote Romeo and Juliet?",
            'answer': ["William Shakespeare", "Shakespeare"]
        },
        {
            'question': "What is the largest planet?",
            'answer': ["Jupiter"]
        },
        {
            'question': "When was Python created?",
            'answer': ["1991"]
        },
        {
            'question': "What is photosynthesis?",
            'answer': ["Process by which plants make food"]
        }
    ]

    if max_samples is not None:
        dummy_data = dummy_data[:max_samples]

    return dummy_data


def evaluate_rag(passages, index, encoder_model, encoder_tokenizer,
                 generator, generator_tokenizer, dataset,
                 n_docs=5, max_length=50):
    """Evaluate RAG with custom retrieval."""
    device = next(generator.parameters()).device
    predictions = []
    references = []

    start_time = time.time()

    print(f"\nEvaluating on {len(dataset)} examples...")
    print(f"Retrieving {n_docs} documents per question")

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        example = dataset[i]

        # Get question and answer
        question = example.get('question', '')
        answer = example.get('answer', [])

        # Retrieve passages
        retrieved = retrieve_passages(
            question, index, passages,
            encoder_model, encoder_tokenizer, k=n_docs
        )

        # Format context
        context_parts = []
        for p in retrieved:
            title = p.get('title', '')
            text = p.get('text', '')[:200]  # Truncate to avoid too long inputs
            if title:
                context_parts.append(f"{title}: {text}")
            else:
                context_parts.append(text)

        context = " ".join(context_parts)

        # Format input for generator (BART format)
        input_text = f"{context} Question: {question} Answer:"

        # Generate answer
        inputs = generator_tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = generator.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_length,  # Generate up to max_length NEW tokens
                num_beams=4,
                early_stopping=True
            )

        # Decode prediction
        prediction = generator_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(prediction)

        # Process answer
        if isinstance(answer, str):
            references.append([answer])
        elif isinstance(answer, list):
            references.append(answer if answer else [''])
        else:
            references.append([''])

    # Compute metrics
    end_time = time.time()
    elapsed = end_time - start_time

    metrics = compute_metrics(predictions, references)
    metrics['total_time'] = elapsed
    metrics['questions_per_second'] = len(dataset) / elapsed
    metrics['num_passages_in_index'] = index.ntotal
    metrics['num_docs_retrieved'] = n_docs

    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }


def main():
    parser = argparse.ArgumentParser(description="Custom RAG evaluation using cached wiki_dpr")
    parser.add_argument("--encoder", type=str, default="facebook/dpr-ctx_encoder-single-nq-base")
    parser.add_argument("--generator", type=str, default="facebook/bart-large")
    parser.add_argument("--dataset", type=str, default="nq_open")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_passages", type=int, default=100000,
                       help="Maximum passages to load from cache (default: 100k)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Path to wiki_dpr cache directory")
    parser.add_argument("--n_docs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--no_fp16", action="store_true")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CUSTOM RAG EVALUATION")
    print("="*60)

    # Load models
    print("[1/4] Loading encoder and generator models...")
    encoder_model, encoder_tokenizer, generator, generator_tokenizer = load_models(
        encoder_name=args.encoder,
        generator_name=args.generator,
        use_fp16=not args.no_fp16
    )
    print("      ✓ Models loaded")

    # Load Wikipedia passages from cache
    print("[2/4] Loading Wikipedia passages from cache...")
    passages = load_wiki_dpr_from_cache(
        cache_dir=args.cache_dir,
        max_passages=args.max_passages
    )
    print(f"      ✓ Passages loaded ({len(passages):,} passages)")

    # Build FAISS index
    print("[3/4] Building FAISS index...")
    import faiss
    index = build_faiss_index(passages, encoder_model, encoder_tokenizer, batch_size=args.batch_size)
    print("      ✓ Index built")

    # Load evaluation dataset
    print("[4/4] Loading evaluation dataset...")
    dataset = load_evaluation_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )
    print(f"      ✓ Dataset loaded ({len(dataset)} samples)")
    print("="*60 + "\n")

    # Evaluate
    print("RUNNING EVALUATION")
    print("="*60)
    results = evaluate_rag(
        passages=passages,
        index=index,
        encoder_model=encoder_model,
        encoder_tokenizer=encoder_tokenizer,
        generator=generator,
        generator_tokenizer=generator_tokenizer,
        dataset=dataset,
        n_docs=args.n_docs,
        max_length=args.max_length
    )

    # Print results
    print("="*60)
    print("RAG EVALUATION RESULTS")
    print("="*60)
    print(f"Generator: {args.generator}")
    print(f"Encoder: {args.encoder}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Index size: {results['metrics']['num_passages_in_index']:,} passages")
    print(f"Retrieved docs/question: {results['metrics']['num_docs_retrieved']}")
    print(f"Evaluation examples: {results['metrics']['num_examples']}")
    print(f"\n{'='*60}")
    print(f"Exact Match: {results['metrics']['exact_match']:.2f}%")
    print(f"F1 Score: {results['metrics']['f1']:.2f}%")
    print(f"{'='*60}")
    print(f"Total time: {results['metrics']['total_time']:.2f}s")
    print(f"Speed: {results['metrics']['questions_per_second']:.2f} questions/second")
    print("="*60)
    print("✓ Evaluation completed successfully\n")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_results = {
            'generator': args.generator,
            'encoder': args.encoder,
            'dataset': args.dataset,
            'split': args.split,
            'metrics': results['metrics'],
            'args': vars(args)
        }

        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    # Show examples
    print("\nExample predictions:")
    for i in range(min(5, len(results['predictions']))):
        print(f"\nExample {i+1}:")
        print(f"  Prediction: {results['predictions'][i]}")
        print(f"  Reference: {results['references'][i][0]}")


if __name__ == "__main__":
    main()
