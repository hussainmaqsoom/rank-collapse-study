"""
extract_attention.py
Extract attention matrices from ALL layers of multiple transformer models.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("Done.")
    return tokenizer, model


def get_attention_all_layers(text, tokenizer, model):
    """
    Extract attention matrices from ALL layers for a given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.attentions


def compute_rank_and_gap(attention_tensor, epsilon=1e-3):
    """
    Given attention tensor for one layer (num_heads, seq_len, seq_len),
    compute average numerical rank and spectral gap across all heads.
    """
    num_heads = attention_tensor.shape[1]
    ranks = []
    gaps = []
    
    for head in range(num_heads):
        matrix = attention_tensor[0, head]
        
        U, S, V = torch.linalg.svd(matrix)
        
        threshold = epsilon * S[0]
        rank = (S > threshold).sum().item()
        ranks.append(rank)
        
        gap = (S[0] / S[1]).item()
        gaps.append(gap)
    
    avg_rank = sum(ranks) / len(ranks)
    avg_gap = sum(gaps) / len(gaps)
    
    return avg_rank, avg_gap


def process_text(text, tokenizer, model):
    """Get average rank and gap across all layers for one text."""
    all_attentions = get_attention_all_layers(text, tokenizer, model)
    
    num_layers = len(all_attentions)
    ranks_per_layer = []
    gaps_per_layer = []
    
    for layer_idx in range(num_layers):
        attn = all_attentions[layer_idx]
        avg_rank, avg_gap = compute_rank_and_gap(attn)
        ranks_per_layer.append(avg_rank)
        gaps_per_layer.append(avg_gap)
    
    return ranks_per_layer, gaps_per_layer


def test_model(model_name, texts):
    """Test one model on multiple texts and average results."""
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    all_ranks = []
    all_gaps = []
    
    for i, text in enumerate(texts):
        if i % 5 == 0:
            print(f"  Processing text {i+1}/{len(texts)}...")
        ranks, gaps = process_text(text, tokenizer, model)
        all_ranks.append(ranks)
        all_gaps.append(gaps)
    
    # Average across texts
    num_layers = len(all_ranks[0])
    final_ranks = []
    final_gaps = []
    
    for layer_idx in range(num_layers):
        layer_ranks = [r[layer_idx] for r in all_ranks]
        layer_gaps = [g[layer_idx] for g in all_gaps]
        final_ranks.append(sum(layer_ranks) / len(layer_ranks))
        final_gaps.append(sum(layer_gaps) / len(layer_gaps))
    
    return final_ranks, final_gaps


if __name__ == "__main__":
    # Test sentences
    texts = [
        "The cat sat on the mat.",
        "In 1492, Christopher Columbus sailed across the Atlantic Ocean.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "Water boils at 100 degrees Celsius at standard pressure.",
        "Photosynthesis converts light energy into chemical energy.",
        "The Earth orbits the Sun at an average distance of 93 million miles.",
        "Shakespeare wrote many plays including Hamlet and Macbeth.",
        "Python is a popular programming language for data science.",
        "The mitochondria is the powerhouse of the cell.",
    ]
    
    # Test GPT-2 and DistilGPT2 and facebook/opt-125m
    models_to_test = ["gpt2", "distilgpt2", "facebook/opt-125m"]
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n=== {model_name.upper()} ===")
        ranks, gaps = test_model(model_name, texts)
        
        print(f"\n{'Layer':<<8} {'Avg Rank':<<12} {'Avg Gap':<<12}")
        print("-" * 32)
        for i, (r, g) in enumerate(zip(ranks, gaps)):
            print(f"{i:<8} {r:<12.2f} {g:<12.4f}")
        
        all_results[model_name] = {
            "ranks": ranks,
            "gaps": gaps,
            "num_layers": len(ranks)
        }
    
    # SAVE RESULTS - THIS IS INSIDE THE MAIN BLOCK
    with open("results/all_models_rank_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to results/all_models_rank_results.json")