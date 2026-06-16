"""
ks_table.py
Generate a publication-ready table of KS test results.
"""

import json


def generate_table():
    with open("results/all_models_rank_results.json", "r") as f:
        results = json.load(f)
    
    # This is a simple text table for now
    # In the paper, you'll format this properly
    print("Table 1: Kolmogorov-Smirnov Test Results for Rank Distributions")
    print("=" * 70)
    print(f"{'Model Pair':<<35} {'KS Statistic':<<15} {'p-value':<<12} {'Result'}")
    print("-" * 70)
    
    # You would compute these from the actual test
    # For now, using the values you obtained
    tests = [
        ("GPT-2 vs. DistilGPT2", 0.3333, 0.7596),
        ("GPT-2 vs. OPT-125M", 0.1667, 0.9985),
        ("DistilGPT2 vs. OPT-125M", 0.2500, 0.9607),
    ]
    
    for pair, ks, p in tests:
        result = "Similar" if p > 0.05 else "Different"
        print(f"{pair:<35} {ks:<15.4f} {p:<12.4f} {result}")
    
    print("=" * 70)
    print("\nNote: H0: distributions are identical. Significance level α = 0.05.")
    print("All p-values > 0.05, failing to reject H0. Rank distributions are statistically similar.")


if __name__ == "__main__":
    generate_table()