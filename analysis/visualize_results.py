"""
Create visualizations for RAG model comparison.

Generates plots for presentation and paper.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_all_results():
    """Load all evaluation results from JSON files in results/metrics."""
    results_dir = Path(__file__).parent.parent / "results" / "metrics"
    results = {}

    # Get all JSON files
    json_files = sorted(results_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            filename = json_file.stem  # Remove .json extension

            # Handle reranker_strategies_100.json specially - extract individual strategies
            if filename == "reranker_strategies_100":
                # New format: {strategy: {metrics: {...}}, ...}
                for strategy in ['basic', 'enhanced', 'diversity']:
                    if strategy in data:
                        strategy_name = strategy.capitalize()
                        results[f'Flan-T5-base +\nReranker ({strategy_name})'] = data[strategy]
            # Skip files that are already handled or are strategy files
            elif filename not in ['baseline_100samples', 'baseline_200samples', 'flan_t5_small_100',
                                  'flan_t5_base_100', 'flan_t5_large_4bit_100', 't5_prompt_100',
                                  'rag_fusion_results']:
                # Add any other JSON files with generic naming
                model_name = filename.replace('_', ' ').replace('100', '').replace('200', '').strip()
                model_name = model_name.title()
                results[f'{model_name}'] = data
            # Handle known files with pretty names
            elif filename == 'baseline_100samples':
                results['RAG-BART\n(baseline)'] = data
            elif filename == 'flan_t5_small_100':
                results['Flan-T5-small\n(77M)'] = data
            elif filename == 'flan_t5_base_100':
                results['Flan-T5-base\n(248M)'] = data
            elif filename == 'flan_t5_large_4bit_100':
                results['Flan-T5-large\n(4-bit, 494M)'] = data
            elif filename == 't5_prompt_100':
                results['T5 Prompt\nEngineering'] = data
            elif filename == 'rag_fusion_results':
                results['RAG Fusion\n(RRF)'] = data

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return results


def create_accuracy_comparison(results, output_dir):
    """
    Create bar chart comparing EM and F1 scores.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save plot
    """
    models = list(results.keys())
    em_scores = [results[m]['metrics']['exact_match'] for m in models]
    f1_scores = [results[m]['metrics']['f1'] for m in models]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar([i - width/2 for i in x], em_scores, width, label='Exact Match', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], f1_scores, width, label='F1 Score', alpha=0.8)

    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Comparison: EM and F1 Scores\n100 Samples from Natural Questions',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(max(em_scores), max(f1_scores)) * 1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_speed_vs_accuracy(results, output_dir):
    """
    Create scatter plot showing speed vs accuracy trade-off.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save plot
    """
    models = list(results.keys())
    speeds = [results[m]['metrics']['questions_per_second'] for m in models]
    em_scores = [results[m]['metrics']['exact_match'] for m in models]

    # Extract parameter sizes for bubble sizes
    param_sizes = {
        'RAG-BART\n(baseline)': 515,
        'Flan-T5-small\n(77M)': 77,
        'Flan-T5-base\n(248M)': 248,
        'Flan-T5-large\n(4-bit, 494M)': 494,
        'Flan-T5-base +\nReranker (Basic)': 281,
        'Flan-T5-base +\nReranker (Enhanced)': 281,
        'Flan-T5-base +\nReranker (Diversity)': 281
    }
    sizes = [param_sizes.get(m, 100) for m in models]

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(speeds, em_scores, s=[s*2 for s in sizes], alpha=0.6,
                        c=range(len(models)), cmap='viridis', edgecolors='black', linewidth=2)

    # Add labels for each point
    for i, model in enumerate(models):
        ax.annotate(model, (speeds[i], em_scores[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    ax.set_xlabel('Speed (questions/second)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Exact Match (%)', fontsize=14, fontweight='bold')
    ax.set_title('Speed vs Accuracy Trade-off\nBubble size = Model parameters',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')  # Log scale for speed
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'speed_vs_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_speedup_comparison(results, output_dir):
    """
    Create bar chart showing speedup relative to baseline.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save plot
    """
    baseline_speed = results['RAG-BART\n(baseline)']['metrics']['questions_per_second']

    models = []
    speedups = []
    colors = []

    for model, res in results.items():
        if model != 'RAG-BART\n(baseline)':
            speed = res['metrics']['questions_per_second']
            speedup = speed / baseline_speed
            models.append(model)
            speedups.append(speedup)
            # Color based on speedup magnitude
            if speedup > 40:
                colors.append('green')
            elif speedup > 10:
                colors.append('orange')
            else:
                colors.append('red')

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(models, speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_xlabel('Speedup Factor (relative to baseline)', fontsize=14, fontweight='bold')
    ax.set_title('Speed Improvement Over RAG-BART Baseline\n100 Samples from Natural Questions',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Baseline (1x)')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
               f'{width:.1f}x faster',
               ha='left', va='center', fontsize=12, fontweight='bold')

    ax.legend(fontsize=12)
    plt.tight_layout()
    output_path = output_dir / 'speedup_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_f1_em_gap_analysis(results, output_dir):
    """
    Create bar chart showing F1 vs EM gap.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save plot
    """
    models = list(results.keys())
    em_scores = [results[m]['metrics']['exact_match'] for m in models]
    f1_scores = [results[m]['metrics']['f1'] for m in models]
    gaps = [f1 - em for f1, em in zip(f1_scores, em_scores)]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['red' if gap < 1 else 'blue' for gap in gaps]
    bars = ax.bar(models, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_ylabel('F1 - EM Gap (%)', fontsize=14, fontweight='bold')
    ax.set_title('F1 vs Exact Match Gap Analysis\nHigher gap = More partial matches',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=11, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='No partial matches (all or nothing)'),
        Patch(facecolor='blue', alpha=0.7, label='Generates partial matches')
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'f1_em_gap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_combined_summary(results, output_dir):
    """
    Create combined plot with multiple subplots.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RAG Model Comparison Summary\n100 Samples from Natural Questions',
                 fontsize=18, fontweight='bold', y=0.995)

    models = list(results.keys())
    em_scores = [results[m]['metrics']['exact_match'] for m in models]
    f1_scores = [results[m]['metrics']['f1'] for m in models]
    speeds = [results[m]['metrics']['questions_per_second'] for m in models]

    # Dynamic parameter sizes based on loaded models
    param_size_map = {
        'RAG-BART\n(baseline)': 515,
        'Flan-T5-small\n(77M)': 77,
        'Flan-T5-base\n(248M)': 248,
        'Flan-T5-large\n(4-bit, 494M)': 494,
        'Flan-T5-base +\nReranker (Basic)': 281,
        'Flan-T5-base +\nReranker (Enhanced)': 281,
        'Flan-T5-base +\nReranker (Diversity)': 281,
        'RAG Fusion\n(RRF)': 281,
        'T5 Prompt\nEngineering': 248
    }
    param_sizes = [param_size_map.get(m, 100) for m in models]

    # 1. Accuracy comparison
    x = range(len(models))
    width = 0.35
    ax1.bar([i - width/2 for i in x], em_scores, width, label='EM', alpha=0.8)
    ax1.bar([i + width/2 for i in x], f1_scores, width, label='F1', alpha=0.8)
    ax1.set_ylabel('Score (%)', fontweight='bold')
    ax1.set_title('Accuracy Scores', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend()

    # 2. Speed comparison
    ax2.bar(models, speeds, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_ylabel('Questions/second', fontweight='bold')
    ax2.set_title('Inference Speed', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    for i, v in enumerate(speeds):
        ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

    # 3. Speed vs Accuracy
    scatter = ax3.scatter(speeds, em_scores, s=[s*2 for s in param_sizes],
                         alpha=0.6, c=range(len(models)), cmap='viridis',
                         edgecolors='black', linewidth=2)
    ax3.set_xlabel('Speed (q/s)', fontweight='bold')
    ax3.set_ylabel('Exact Match (%)', fontweight='bold')
    ax3.set_title('Speed vs Accuracy Trade-off', fontweight='bold', fontsize=14)
    ax3.set_xscale('log')

    # 4. Speedup
    baseline_speed = results['RAG-BART\n(baseline)']['metrics']['questions_per_second']
    speedups = [speed / baseline_speed for speed in speeds[1:]]  # Exclude baseline
    model_names = models[1:]
    ax4.barh(model_names, speedups, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Speedup Factor', fontweight='bold')
    ax4.set_title('Speedup vs Baseline', fontweight='bold', fontsize=14)
    ax4.axvline(x=1, color='red', linestyle='--', linewidth=2)
    for i, v in enumerate(speedups):
        ax4.text(v + 1, i, f'{v:.1f}x', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'combined_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main function."""
    print("Loading evaluation results...")
    results = load_all_results()

    if not results:
        print("Error: No results found!")
        return

    print(f"Loaded {len(results)} model results\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations in {output_dir}...\n")

    # Generate all plots
    create_accuracy_comparison(results, output_dir)
    create_speed_vs_accuracy(results, output_dir)
    create_speedup_comparison(results, output_dir)
    create_f1_em_gap_analysis(results, output_dir)
    create_combined_summary(results, output_dir)

    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. accuracy_comparison.png - EM and F1 scores")
    print("  2. speed_vs_accuracy.png - Trade-off analysis")
    print("  3. speedup_comparison.png - Speed improvements")
    print("  4. f1_em_gap.png - Partial match analysis")
    print("  5. combined_summary.png - All metrics together")


if __name__ == "__main__":
    main()