import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_all_results():
    """Load all optimization results from JSON files"""
    results_dir = 'results'
    all_results = []
    
    if not os.path.exists(results_dir):
        print("No results directory found. Please run optimization methods first.")
        return []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                try:
                    result = json.load(f)
                    all_results.append(result)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename}. File might be empty or corrupt. Skipping.")
    
    return all_results

def create_comparison_plots(results):
    """Create comparison visualizations"""
    if not results:
        print("No results to plot.")
        return
    
    methods = [r['method'] for r in results]
    accuracies = [r['metrics']['accuracy'] for r in results]
    f1_scores = [r['metrics']['f1_score'] for r in results]
    times = [r['time_taken'] for r in results]
    iterations = [r['n_iterations'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Optimization Methods Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    axes[0, 0].bar(methods, accuracies, color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Test Accuracy by Method', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    
    # F1 Score comparison
    axes[0, 1].bar(methods, f1_scores, color='darkorange', alpha=0.7)
    axes[0, 1].set_ylabel('F1 Score', fontsize=12)
    axes[0, 1].set_title('Test F1 Score by Method', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    
    # Time comparison
    axes[1, 0].bar(methods, times, color='forestgreen', alpha=0.7)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Execution Time by Method', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(times):
        axes[1, 0].text(i, v + max(times)*0.02, f'{v:.2f}s', ha='center', fontsize=9)
    
    # Efficiency (Accuracy per second)
    efficiency = [acc / (t + 1) for acc, t in zip(accuracies, times)]
    axes[1, 1].bar(methods, efficiency, color='purple', alpha=0.7)
    axes[1, 1].set_ylabel('Accuracy / Time', fontsize=12)
    axes[1, 1].set_title('Efficiency (Accuracy per Second)', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(efficiency):
        axes[1, 1].text(i, v + max(efficiency)*0.02, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to results/comparison_plot.png")
    plt.close()


def create_summary_table(results):
    """Create a summary table of results"""
    if not results:
        return None
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Method': r['method'],
            'Accuracy': f"{r['metrics']['accuracy']:.4f}",
            'F1 Score': f"{r['metrics']['f1_score']:.4f}",
            'Time (s)': f"{r['time_taken']:.2f}",
            'Iterations': r['n_iterations'],
            'Efficiency': f"{r['metrics']['accuracy'] / (r['time_taken'] + 1):.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    df.to_csv('results/summary_table.csv', index=False)
    print("Summary table saved to results/summary_table.csv")
    
    return df

if __name__ == "__main__":
    results = load_all_results()
    
    if results:
        create_comparison_plots(results)
        create_summary_table(results)
    else:
        print("No results found. Please run the optimization methods first.")