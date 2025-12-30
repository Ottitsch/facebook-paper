"""
Compare results from all RAG model evaluations.

Loads JSON result files and creates comparison tables for presentation.
"""

import json
import pandas as pd
from pathlib import Path


def load_all_results():
    """
    Load all evaluation results from JSON files.

    Returns:
        Dictionary mapping model names to their results
    """
    results_dir = Path(__file__).parent.parent / "results" / "metrics"

    results = {}

    # Load baseline RAG-BART results
    baseline_file = results_dir / "baseline_100samples.json"
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            results['RAG-BART (baseline)'] = json.load(f)

    # Load Flan-T5 results
    flan_files = {
        'Flan-T5-small': results_dir / "flan_t5_small_100.json",
        'Flan-T5-base': results_dir / "flan_t5_base_100.json",
        'Flan-T5-large (4-bit)': results_dir / "flan_t5_large_4bit_100.json"
    }

    for name, filepath in flan_files.items():
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {filepath} not found")

    return results


def extract_model_size(model_name):
    """Extract parameter count from model name."""
    size_map = {
        'RAG-BART (baseline)': 515,
        'Flan-T5-small': 77,
        'Flan-T5-base': 248,
        'Flan-T5-large (4-bit)': 494
    }
    return size_map.get(model_name, 0)


def create_comparison_table(results):
    """
    Create comparison table from results.

    Args:
        results: Dictionary of model results

    Returns:
        pandas DataFrame with comparison
    """
    data = []

    for model_name, res in results.items():
        metrics = res.get('metrics', {})

        row = {
            'Model': model_name,
            'Parameters (M)': extract_model_size(model_name),
            'Exact Match (%)': metrics.get('exact_match', 0),
            'F1 Score (%)': metrics.get('f1', 0),
            'Speed (q/s)': metrics.get('questions_per_second', 0),
            'Samples': metrics.get('num_examples', 0)
        }

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by parameter count
    df = df.sort_values('Parameters (M)')

    return df


def create_speedup_comparison(results):
    """
    Calculate speedup relative to baseline.

    Args:
        results: Dictionary of model results

    Returns:
        pandas DataFrame with speedup analysis
    """
    baseline_speed = results.get('RAG-BART (baseline)', {}).get('metrics', {}).get('questions_per_second', 1)

    data = []

    for model_name, res in results.items():
        metrics = res.get('metrics', {})
        speed = metrics.get('questions_per_second', 0)

        row = {
            'Model': model_name,
            'Speed (q/s)': speed,
            'Speedup vs Baseline': speed / baseline_speed if baseline_speed > 0 else 0,
            'Time for 100 samples (s)': 100 / speed if speed > 0 else float('inf')
        }

        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('Speed (q/s)', ascending=False)

    return df


def create_accuracy_summary(results):
    """
    Create accuracy summary comparing all models.

    Args:
        results: Dictionary of model results

    Returns:
        pandas DataFrame with accuracy metrics
    """
    data = []

    for model_name, res in results.items():
        metrics = res.get('metrics', {})

        row = {
            'Model': model_name,
            'Exact Match (%)': metrics.get('exact_match', 0),
            'F1 Score (%)': metrics.get('f1', 0),
            'F1 - EM Gap (%)': metrics.get('f1', 0) - metrics.get('exact_match', 0)
        }

        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('Exact Match (%)', ascending=False)

    return df


def print_summary(results):
    """Print formatted summary of all results."""
    print("\n" + "="*80)
    print("RAG MODEL COMPARISON - 100 SAMPLE EVALUATION")
    print("="*80)

    # Main comparison table
    print("\n1. OVERALL COMPARISON")
    print("-" * 80)
    comparison = create_comparison_table(results)
    print(comparison.to_string(index=False))

    # Speed analysis
    print("\n\n2. SPEED ANALYSIS")
    print("-" * 80)
    speedup = create_speedup_comparison(results)
    print(speedup.to_string(index=False))

    # Accuracy analysis
    print("\n\n3. ACCURACY ANALYSIS")
    print("-" * 80)
    accuracy = create_accuracy_summary(results)
    print(accuracy.to_string(index=False))

    # Key findings
    print("\n\n4. KEY FINDINGS")
    print("-" * 80)

    # Find best model for each category
    best_accuracy = max(results.items(), key=lambda x: x[1]['metrics']['exact_match'])
    best_speed = max(results.items(), key=lambda x: x[1]['metrics']['questions_per_second'])

    print(f"• Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['metrics']['exact_match']:.1f}% EM)")
    print(f"• Fastest: {best_speed[0]} ({best_speed[1]['metrics']['questions_per_second']:.2f} q/s)")

    # Find best balance (similar accuracy to baseline but faster)
    baseline_em = results.get('RAG-BART (baseline)', {}).get('metrics', {}).get('exact_match', 0)
    baseline_speed = results.get('RAG-BART (baseline)', {}).get('metrics', {}).get('questions_per_second', 0)

    for name, res in results.items():
        if name != 'RAG-BART (baseline)':
            em = res['metrics']['exact_match']
            speed = res['metrics']['questions_per_second']

            # Check if EM within 5% of baseline but much faster
            if abs(em - baseline_em) <= 5 and speed > baseline_speed * 2:
                speedup = speed / baseline_speed
                print(f"• Best Balance: {name}")
                print(f"  - Accuracy: {em:.1f}% EM (baseline: {baseline_em:.1f}%)")
                print(f"  - Speed: {speedup:.1f}x faster than baseline")

    print("\n" + "="*80)


def save_comparison_csv(results, output_file):
    """
    Save comparison table to CSV.

    Args:
        results: Dictionary of model results
        output_file: Path to save CSV
    """
    comparison = create_comparison_table(results)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_path, index=False)
    print(f"\nComparison table saved to: {output_path}")


def main():
    """Main function."""
    print("Loading evaluation results...")
    results = load_all_results()

    if not results:
        print("Error: No results found!")
        print("Please run evaluations first:")
        print("  - experiments/eval_rag_baseline.py --max_samples 100")
        print("  - experiments/eval_rag_flan_t5.py --model_name google/flan-t5-small --max_samples 100")
        print("  - experiments/eval_rag_flan_t5.py --model_name google/flan-t5-base --max_samples 100")
        print("  - experiments/eval_rag_flan_t5.py --model_name google/flan-t5-large --max_samples 100 --use_4bit")
        return

    print(f"Loaded {len(results)} model results")

    # Print summary
    print_summary(results)

    # Save CSV
    output_file = Path(__file__).parent.parent / "results" / "comparison_table.csv"
    save_comparison_csv(results, output_file)


if __name__ == "__main__":
    main()
