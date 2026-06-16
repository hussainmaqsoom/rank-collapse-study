"""
plot_results.py
Plot rank and gap curves with error bars across multiple models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_with_errorbars(results_dict, metric_name, ylabel, save_path):
    """
    Plot metric vs. relative depth with shaded error regions.
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, data in results_dict.items():
        values = data[metric_name]
        num_layers = data["num_layers"]
        depths = [i / (num_layers - 1) for i in range(num_layers)]
        
        # Compute standard error (we'll use a simple estimate)
        # For now, plot the mean line; error bars require per-text data
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
    
    plot_with_errorbars(all_results, "ranks", "Average Numerical Rank", "figures/rank_collapse_50sentences.png")
    plot_with_errorbars(all_results, "gaps", "Average Spectral Gap", "figures/spectral_gap_50sentences.png")