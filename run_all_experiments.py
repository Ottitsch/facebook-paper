#!/usr/bin/env python
"""
Comprehensive Project Runner - Executes ALL evaluation pipelines.
Runs RAG Baseline, Flan-T5 variants, custom RAG, and prompt-based T5.
Aggregates results and generates comparison report.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def print_section(title, char="="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {title}".ljust(width - 1) + char)
    print(f"{char * width}\n")


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def run_command(cmd: str, description: str) -> Tuple[bool, str]:
    """
    Run a shell command and return success status and output.
    Shows real-time progress output.
    
    Args:
        cmd: Command to run
        description: Description for logging
    
    Returns:
        Tuple of (success: bool, output: str)
    """
    print(f"▶ {description}")
    print(f"  Command: {cmd}\n")
    print("  Progress: ", end="", flush=True)
    
    try:
        # Use Popen to show real-time output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output)
                # Show dots for progress
                if 'Evaluating' in output or '%' in output or 'example' in output.lower():
                    print(".", end="", flush=True)
        
        returncode = process.wait()
        output_text = ''.join(output_lines)
        
        if returncode == 0:
            print(f"\n✓ {description} completed successfully\n")
            return True, output_text
        else:
            error = process.stderr.read() if process.stderr else ""
            print(f"\n✗ {description} failed")
            print(f"  Error: {error}\n")
            return False, error
    
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} timed out (>1 hour)\n")
        return False, "Timeout"
    except Exception as e:
        print(f"\n✗ {description} error: {e}\n")
        return False, str(e)


def ensure_venv():
    """Ensure virtual environment exists and is activated."""
    venv_path = Path("venv311")
    if not venv_path.exists():
        print("⚠ Virtual environment not found!")
        print("  Creating: py -3.11 -m venv venv311")
        run_command("py -3.11 -m venv venv311", "Create virtual environment")


def check_environment():
    """Check and print environment information."""
    print_section("Environment Check")
    
    python_cmd = "venv311\\Scripts\\python.exe"
    
    # Python version
    success, output = run_command(f"{python_cmd} --version", "Check Python version")
    if success:
        print(f"  Python version: {output.strip()}")
    
    # PyTorch/CUDA
    import_cmd = f"{python_cmd} -c \"import torch; print(f'PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"N/A\\\"}}')\""
    success, output = run_command(import_cmd, "Check PyTorch/CUDA")
    if success:
        print(f"  {output.strip()}")


def check_dataset():
    """Check if NQ-Open dataset is available."""
    print_section("Dataset Check")
    
    data_dir = Path("data/datasets--google-research-datasets--nq_open/snapshots")
    
    if data_dir.exists():
        snapshots = list(data_dir.glob("*/"))
        if snapshots:
            snapshot = snapshots[0]
            nq_dir = snapshot / "nq_open"
            if nq_dir.exists():
                parquet_files = list(nq_dir.glob("*.parquet"))
                print(f"✓ Dataset found")
                print(f"  Location: {snapshot}")
                print(f"  Files: {len(parquet_files)} parquet files\n")
                
                for f in parquet_files:
                    size_mb = f.stat().st_size / 1e6
                    print(f"    • {f.name} ({size_mb:.1f} MB)")
                
                print()
                return True
    
    print("✗ Dataset not found!")
    print("  Run: venv311\\Scripts\\activate && python scripts/download_nq_hf_hub.py\n")
    return False


def run_baseline_rag(max_samples: int = 100) -> bool:
    """Run baseline RAG-BART evaluation."""
    print_subsection("Baseline RAG-BART (facebook/rag-sequence-nq)")
    
    cmd = (
        f"venv311\\Scripts\\activate && python experiments\\eval_rag_baseline.py "
        f"--max_samples {max_samples} "
        f"--output_file results/metrics/baseline_{max_samples}samples.json"
    )
    
    success, _ = run_command(cmd, f"Baseline RAG evaluation ({max_samples} samples)")
    return success


def run_flan_t5_variants(max_samples: int = 100) -> bool:
    """Run Flan-T5 variants (small, base, large with 4-bit)."""
    print_subsection("Flan-T5 Variants")
    
    variants = [
        ("google/flan-t5-small", "flan_t5_small", False),
        ("google/flan-t5-base", "flan_t5_base", False),
        ("google/flan-t5-large", "flan_t5_large_4bit", True),
    ]
    
    all_success = True
    
    for model_name, output_name, use_4bit in variants:
        print_subsection(f"  → {model_name}")
        
        cmd = (
            f"venv311\\Scripts\\activate && python experiments\\eval_rag_flan_t5.py "
            f"--model_name {model_name} "
            f"--max_samples {max_samples} "
        )
        
        if use_4bit:
            cmd += "--use_4bit "
        
        cmd += f"--output_file results/metrics/{output_name}_{max_samples}.json"
        
        success, _ = run_command(
            cmd,
            f"Flan-T5 evaluation: {model_name} ({max_samples} samples)"
        )
        
        all_success = all_success and success
    
    return all_success


def run_flan_t5_reranked(max_samples: int = 100) -> bool:
    """Run Flan-T5 with reranking."""
    print_subsection("Flan-T5 with Reranking (Cross-Encoder)")
    
    cmd = (
        f"venv311\\Scripts\\activate && python experiments\\eval_rag_flan_t5_reranked.py "
        f"--model_name google/flan-t5-base "
        f"--max_samples {max_samples} "
        f"--retrieval_k 50 "
        f"--rerank_top_k 15 "
        f"--output_file results/metrics/flan_t5_reranked_{max_samples}.json"
    )
    
    success, _ = run_command(cmd, f"Flan-T5 with reranking ({max_samples} samples)")
    return success


def run_reranker_strategies(max_samples: int = 100) -> bool:
    """Run reranker strategy comparison (Enhanced vs Diversity) and save results."""
    print_subsection("Reranker Strategy Comparison (Enhanced vs Diversity)")
    
    # Import here to avoid issues at module load time
    import json

    import pandas as pd

    from experiments.eval_rag_flan_t5 import (load_flan_t5_with_retriever,
                                              load_natural_questions)
    from experiments.eval_rag_flan_t5_reranked import evaluate_flan_t5_reranked
    from experiments.reranker import load_reranker
    
    try:
        print("Loading models and data...")
        model, tokenizer, retriever, question_encoder, question_encoder_tokenizer = load_flan_t5_with_retriever(
            model_name="google/flan-t5-base",
            use_fp16=True,
            n_docs=50,
            use_4bit=False
        )
        
        dataset = load_natural_questions(split="validation", max_samples=max_samples)
        
        results_by_strategy = {}
        
        # Test each strategy
        for strategy in ["basic", "enhanced", "diversity"]:
            print(f"\nTesting Strategy: {strategy.upper()}")
            
            # Load reranker with specific strategy
            reranker = load_reranker(strategy=strategy, use_fp16=True)
            
            # Run evaluation
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
            
            # Store results
            results_by_strategy[strategy] = {
                'metrics': {
                    'exact_match': metrics['exact_match'],
                    'f1': metrics['f1'],
                    'questions_per_second': metrics['questions_per_second'],
                    'total_time': metrics['total_time'],
                    'num_examples': max_samples
                }
            }
            
            print(f"{strategy.upper()} Results:")
            print(f"  EM: {metrics['exact_match']:.2f}%")
            print(f"  F1: {metrics['f1']:.2f}%")
            print(f"  Speed: {metrics['questions_per_second']:.2f} q/s")
            print(f"  Total Time: {metrics['total_time']:.1f}s")
        
        # Save JSON results in standard format matching other metric files
        # Structure: {strategy: {metrics: {...}}, ...}
        json_results = results_by_strategy
        
        json_path = Path(f"results/metrics/reranker_strategies_{max_samples}.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n✓ JSON results saved to: {json_path}")
        
        # Append to comparison_all.csv
        comparison_rows = []
        for strategy, metrics in results_by_strategy.items():
            comparison_rows.append({
                'Model': f'Flan-T5-base + Reranker ({strategy.capitalize()})',
                'EM': f"{metrics['exact_match']:.1f}%",
                'F1': f"{metrics['f1']:.1f}%",
                'Speed (q/s)': f"{metrics['speed']:.2f}",
                'Total Time': f"{metrics['total_time']:.1f}s"
            })
        
        # Load existing comparison_all.csv
        comparison_path = Path("results/comparison_all.csv")
        if comparison_path.exists():
            df_existing = pd.read_csv(comparison_path)
        else:
            df_existing = pd.DataFrame()
        
        # Check if models already exist and remove them to avoid duplicates
        for row in comparison_rows:
            df_existing = df_existing[~df_existing['Model'].str.contains(row['Model'], na=False)]
        
        # Append new rows
        df_new_rows = pd.DataFrame(comparison_rows)
        df_combined = pd.concat([df_existing, df_new_rows], ignore_index=True)
        
        # Save
        df_combined.to_csv(comparison_path, index=False)
        
        print(f"✓ Results appended to: {comparison_path}")
        print("\nUpdated comparison_all.csv:")
        print(df_combined.to_string())
        
        return True
        
    except Exception as e:
        print(f"✗ Reranker comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_t5_prompt(max_samples: int = 100) -> bool:
    """Run T5 with prompt engineering."""
    print_subsection("T5 with Prompt Engineering")
    
    cmd = (
        f"venv311\\Scripts\\activate && python experiments\\eval_rag_t5_promt.py "
        f"--max_samples {max_samples} "
        f"--output_file results/metrics/t5_prompt_{max_samples}.json"
    )
    
    success, _ = run_command(cmd, f"T5 prompt engineering ({max_samples} samples)")
    return success


def load_result_file(filepath: Path) -> dict:
    """Load a result JSON file safely."""
    try:
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"  ⚠ Could not load {filepath}: {e}")
    
    return {}


def compare_results(max_samples: int = 100):
    """Load and compare all results."""
    print_section("Results Comparison")
    
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected result files
    result_files = {
        "RAG-BART": f"baseline_{max_samples}samples.json",
        "Flan-T5-small": f"flan_t5_small_{max_samples}.json",
        "Flan-T5-base": f"flan_t5_base_{max_samples}.json",
        "Flan-T5-large (4-bit)": f"flan_t5_large_4bit_{max_samples}.json",
        "Flan-T5 with Reranking": f"flan_t5_reranked_{max_samples}.json",
        "T5 Prompt Engineering": f"t5_prompt_{max_samples}.json",
    }
    
    all_results = {}
    
    print("Loading results...\n")
    
    for model_name, filename in result_files.items():
        filepath = results_dir / filename
        
        if filepath.exists():
            result = load_result_file(filepath)
            if result and 'metrics' in result:
                all_results[model_name] = result['metrics']
                print(f"✓ {model_name}: Loaded")
            else:
                print(f"⚠ {model_name}: File found but invalid format")
        else:
            print(f"✗ {model_name}: File not found ({filename})")
    
    if not all_results:
        print("\n⚠ No results loaded. Check if evaluations ran successfully.")
        return
    
    # Print comparison table
    print_section("Performance Metrics Comparison")
    
    # Collect metrics
    metrics_data = []
    
    for model_name in sorted(all_results.keys()):
        metrics = all_results[model_name]
        
        em = metrics.get('exact_match', 0)
        f1 = metrics.get('f1', 0)
        speed = metrics.get('questions_per_second', 0)
        
        metrics_data.append({
            'Model': model_name,
            'EM': f"{em:.1f}%",
            'F1': f"{f1:.1f}%",
            'Speed (q/s)': f"{speed:.2f}",
            'Total Time': f"{metrics.get('total_time', 0):.1f}s"
        })
    
    # Print table header
    print(f"{'Model':<30} {'EM':<10} {'F1':<10} {'Speed':<12} {'Time':<10}")
    print("-" * 72)
    
    # Print table rows
    for row in metrics_data:
        print(
            f"{row['Model']:<30} {row['EM']:<10} {row['F1']:<10} "
            f"{row['Speed (q/s)']:<12} {row['Total Time']:<10}"
        )
    
    # Save comparison to CSV
    import csv
    
    csv_path = Path("results/comparison_all.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'EM', 'F1', 'Speed (q/s)', 'Total Time'])
        writer.writeheader()
        writer.writerows(metrics_data)
    
    print(f"\n✓ Comparison saved to: {csv_path}\n")
    
    # Generate summary statistics
    print_section("Summary Statistics")
    
    if all_results:
        ems = [all_results[m].get('exact_match', 0) for m in all_results]
        speeds = [all_results[m].get('questions_per_second', 0) for m in all_results if all_results[m].get('questions_per_second', 0) > 0]
        
        print(f"Best Exact Match:     {max(ems):.1f}% ({[k for k,v in all_results.items() if v.get('exact_match')==max(ems)][0]})")
        print(f"Best F1 Score:        {max([all_results[m].get('f1', 0) for m in all_results]):.1f}%")
        print(f"Fastest Model:        {max(speeds):.2f} q/s ({[k for k,v in all_results.items() if v.get('questions_per_second')==max(speeds)][0]})")
        print(f"Total Models Tested:  {len(all_results)}\n")


def main():
    """Main runner orchestration."""
    
    # Parse command line arguments
    max_samples = 100
    pipelines = None
    
    if len(sys.argv) > 1:
        try:
            max_samples = int(sys.argv[1])
        except ValueError:
            pass
    
    # Check for pipeline selection: --only-reranker, --only-baseline, etc.
    if "--only-reranker" in sys.argv:
        pipelines = ["Reranker Strategies"]
    elif "--only-baseline" in sys.argv:
        pipelines = ["Baseline RAG"]
    elif "--only-flan" in sys.argv:
        pipelines = ["Flan-T5 Variants"]
    elif "--only-prompt" in sys.argv:
        pipelines = ["T5 Prompt Engineering"]
    
    try:
        print_section("RAG PROJECT - COMPREHENSIVE EVALUATION RUNNER", "█")
        print(f"Max samples per evaluation: {max_samples}")
        print(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Show expected timing
        print("Expected Timing Per Pipeline:")
        print("  • Baseline RAG-BART:     ~15-20 minutes (slowest)")
        print("  • Flan-T5-small:         ~20-30 seconds")
        print("  • Flan-T5-base:          ~25-35 seconds")
        print("  • Flan-T5-large (4-bit): ~1-2 minutes")
        print("  • Flan-T5 + Reranking:   ~2-3 minutes")
        print("  • T5 Prompt Engineering: ~1-2 minutes")
        print(f"  • TOTAL EXPECTED TIME:   ~20-30 minutes\n")
        print("Progress shown as dots (...) as each evaluation processes.\n")
        
        # Ensure environment
        ensure_venv()
        
        # Check environment
        check_environment()
        
        # Check dataset
        if not check_dataset():
            print("\n⚠ Dataset is required to run evaluations.")
            print("  Run: venv311\\Scripts\\activate && python scripts/download_nq_hf_hub.py")
            return 1
        
        # Run all evaluations
        print_section("Running All Evaluations", "█")
        
        all_results = {
            "Baseline RAG": run_baseline_rag,
            "Flan-T5 Variants": run_flan_t5_variants,
            "Flan-T5 with Reranking": run_flan_t5_reranked,
            "Reranker Strategies": run_reranker_strategies,
            "T5 Prompt Engineering": run_t5_prompt,
        }
        
        # Filter pipelines if specified
        if pipelines:
            results = {k: v(max_samples) for k, v in all_results.items() if k in pipelines}
        else:
            results = {k: v(max_samples) for k, v in all_results.items()}
        
        # Print execution summary
        print_section("Execution Summary")
        
        for pipeline, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {pipeline}")
        
        # Compare results
        compare_results(max_samples)
        
        # Final summary
        print_section("Project Complete", "█")
        print(f"All evaluations completed at {datetime.now().isoformat()}")
        print("Results saved to: results/metrics/")
        print("Comparison file: results/comparison_all.csv\n")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print_section("Error")
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
