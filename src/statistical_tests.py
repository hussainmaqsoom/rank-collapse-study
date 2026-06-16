import json
from scipy.stats import ks_2samp

with open("results/all_models_rank_results.json", "r") as f:
    results = json.load(f)

models = list(results.keys())

for i in range(len(models)):
    for j in range(i+1, len(models)):
        a, b = models[i], models[j]
        stat, p = ks_2samp(results[a]["ranks"], results[b]["ranks"])
        print(f"{a} vs {b}: KS={stat:.4f}, p={p:.4f}, {'SIMILAR' if p > 0.05 else 'DIFFERENT'}")