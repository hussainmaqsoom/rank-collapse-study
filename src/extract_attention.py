"""
extract_attention.py
Extract attention matrices from transformer models.
"""

import torch
from transformers import GPT2Tokenizer, GPT2Model


def load_model_and_tokenizer(model_name="gpt2"):
    """
    Load a pretrained model and its tokenizer.
    Returns: tokenizer, model
    """
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float32
    )
    model.eval()  # Set to evaluation mode (no training)
    print("Done.")
    return tokenizer, model


def get_attention_matrix(text, tokenizer, model, layer=0, head=0):
    """
    Given text, return the attention matrix for a specific layer and head.
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract attention from specified layer and head
    # outputs.attentions is a tuple of length num_layers
    # Each element has shape: (batch_size, num_heads, seq_len, seq_len)
    attention = outputs.attentions[layer]  # Get layer
    attention_matrix = attention[0, head]    # Get first batch item, specified head
    
    return attention_matrix


def compute_svd_and_rank(matrix, epsilon=1e-3):
    """
    Compute SVD and numerical rank of a matrix.
    Returns: singular_values, numerical_rank, spectral_gap
    """
    # SVD: matrix = U * diag(S) * V^T
    U, S, V = torch.linalg.svd(matrix)
    
    # Numerical rank: count singular values above threshold
    threshold = epsilon * S[0]  # epsilon times largest singular value
    numerical_rank = (S > threshold).sum().item()
    
    # Spectral gap: ratio of largest to second largest
    spectral_gap = (S[0] / S[1]).item()
    
    return S, numerical_rank, spectral_gap


if __name__ == "__main__":
    # Test with a simple sentence
    text = "The cat sat on the mat."
    
    # Load model
    tokenizer, model = load_model_and_tokenizer("gpt2")
    
    # Get attention from layer 0, head 0
    attn_matrix = get_attention_matrix(text, tokenizer, model, layer=0, head=0)
    
    print(f"\nAttention matrix shape: {attn_matrix.shape}")
    print(f"Sequence length: {attn_matrix.shape[0]}")
    
    # Compute SVD and rank
    singular_values, rank, gap = compute_svd_and_rank(attn_matrix)
    
    print(f"\nSingular values (first 5): {singular_values[:5]}")
    print(f"Numerical rank (eps=1e-3): {rank}")
    print(f"Spectral gap: {gap:.4f}")