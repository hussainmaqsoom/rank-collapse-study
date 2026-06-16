"""
extract_attention.py
Extract attention matrices from ALL layers of a transformer model.
"""

import torch
from transformers import AutoTokenizer, AutoModel


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
    Returns: list of attention tensors, one per layer
    Each tensor shape: (num_heads, seq_len, seq_len)
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions is a tuple of length = num_layers
    # Each element: (batch_size, num_heads, seq_len, seq_len)
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
        matrix = attention_tensor[0, head]  # (seq_len, seq_len)
        
        # SVD
        U, S, V = torch.linalg.svd(matrix)
        
        # Numerical rank
        threshold = epsilon * S[0]
        rank = (S > threshold).sum().item()
        ranks.append(rank)
        
        # Spectral gap
        gap = (S[0] / S[1]).item()
        gaps.append(gap)
    
    # Average across heads
    avg_rank = sum(ranks) / len(ranks)
    avg_gap = sum(gaps) / len(gaps)
    
    return avg_rank, avg_gap


if __name__ == "__main__":
    model_name = "gpt2"
    text = "The cat sat on the mat because it was tired and wanted to rest after a long day of chasing mice and exploring the neighborhood. The sun was setting and the air was cool."
    
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    # Get all layers
    all_attentions = get_attention_all_layers(text, tokenizer, model)
    num_layers = len(all_attentions)
    print(f"\nNumber of layers: {num_layers}")
    
    # Compute rank and gap for each layer
    print(f"\n{'Layer':<<8} {'Avg Rank':<<12} {'Avg Gap':<<12}")
    print("-" * 32)
    
    for layer_idx in range(num_layers):
        attn = all_attentions[layer_idx]  # (1, num_heads, seq_len, seq_len)
        avg_rank, avg_gap = compute_rank_and_gap(attn)
        print(f"{layer_idx:<8} {avg_rank:<12.2f} {avg_gap:<12.4f}")
