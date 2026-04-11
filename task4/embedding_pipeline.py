"""
Task 4: CLIP Text Embedding Pipeline
This module provides a reusable text preprocessing pipeline that
converts natural language text prompts into 512-dimensional semantic
embeddings using the CLIP model from Hugging Face Transformers.

These embeddings are used as conditioning inputs for text-to-image
models in Tasks 5 and 6 of this internship.

Model: openai/clip-vit-base-patch32
Output: L2-normalised 512-dimensional float tensor

Usage:
    from embedding_pipeline import text_to_embedding, texts_to_embeddings

    emb = text_to_embedding("a red circle on black background")
    # Returns: torch.Tensor of shape (1, 512)

    batch = texts_to_embeddings(["a circle", "a square", "a triangle"])
    # Returns: torch.Tensor of shape (3, 512)
"""

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

# Load CLIP tokenizer and text encoder from Hugging Face Hub
# These are loaded once at module import time for efficiency
print("Loading CLIP tokenizer and text encoder...")
tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_encoder.eval()  # set to evaluation mode — disables dropout


def text_to_embedding(text: str) -> torch.Tensor:
    """
    Convert a single text string to a normalised CLIP embedding.

    Pipeline steps:
        1. Input validation — empty/invalid text replaced with default
        2. Truncation — very long inputs capped at 300 characters
        3. Tokenisation — text split into subword tokens (max 77 tokens)
        4. Encoding — tokens passed through CLIP text encoder
        5. Pooling — pooler_output extracts one vector for whole sentence
        6. Normalisation — L2 normalisation ensures unit-length vector

    Args:
        text (str): Input text prompt. Can be any natural language string.

    Returns:
        torch.Tensor: Normalised embedding of shape (1, 512).
                      Values are float32, L2 norm = 1.0.

    Examples:
        >>> emb = text_to_embedding("a photo of sks flower")
        >>> emb.shape
        torch.Size([1, 512])
        >>> emb.norm().item()
        1.0
    """
    # Input validation — replace invalid inputs with a safe default
    if not isinstance(text, str) or not text.strip():
        text = "a flower"  # safe fallback for empty or invalid input

    # Truncate very long inputs to prevent memory issues
    text = text[:300]

    # Tokenise: convert text to token IDs with padding and truncation
    # max_length=77 is CLIP's maximum context length
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    # Encode: pass token IDs through CLIP text encoder
    with torch.no_grad():  # no gradient computation needed for inference
        embedding = text_encoder(**tokens).pooler_output  # shape: (1, 512)

    # L2 normalise: scale vector to unit length for cosine similarity
    return F.normalize(embedding, p=2, dim=1)


def texts_to_embeddings(texts: list) -> torch.Tensor:
    """
    Convert a list of text strings to normalised CLIP embeddings in batch.

    More efficient than calling text_to_embedding() in a loop because
    all texts are processed in a single forward pass through the encoder.

    Args:
        texts (list): List of input text strings.

    Returns:
        torch.Tensor: Normalised embeddings of shape (N, 512)
                      where N = len(texts).

    Examples:
        >>> batch = texts_to_embeddings(["a circle", "a square"])
        >>> batch.shape
        torch.Size([2, 512])
    """
    # Tokenise all texts together with padding to same length
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        embeddings = text_encoder(**tokens).pooler_output  # shape: (N, 512)

    return F.normalize(embeddings, p=2, dim=1)


if __name__ == "__main__":
    # Quick self-test when run directly
    test_prompts = [
        "a photo of sks flower",
        "a red circle on black background",
        "a blue square shape",
    ]
    print("Testing embedding pipeline...")
    for prompt in test_prompts:
        emb = text_to_embedding(prompt)
        print(f"  '{prompt}'")
        print(f"    Shape: {emb.shape}  Norm: {emb.norm().item():.4f}")
    print("All tests passed.")
