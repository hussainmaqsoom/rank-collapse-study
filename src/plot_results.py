"""
plot_results.py
Plot rank collapse and spectral gap curves.
"""

import json
import matplotlib.pyplot as plt


def plot_metric(results_dict, metric_name, ylabel, save_path):
    """
    Plot any metric vs. relative depth for multiple models.
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, data in results_dict.items():
        values = data[metric_name]
        num_layers = data["num_layers"]
        depths = [i / (num_layers - 1) for i in range(num_layers)]
        
        plt.plot(depths, values, marker='o', label=model_name, linewidth=2, markersize=6)
    
    plt.xlabel("Relative Depth (Layer / Total Layers)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel} Across Transformer Architectures", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    with open("results/all_models_rank_results.json", "r") as f:
        all_results = json.load(f)
    
    # Figure 1: Rank
    plot_metric(all_results, "ranks", "Average Numerical Rank", "figures/rank_collapse_comparison.png")
    
    # Figure 2: Spectral Gap
    plot_metric(all_results, "gaps", "Average Spectral Gap", "figures/spectral_gap_comparison.png")