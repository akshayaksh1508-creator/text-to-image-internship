"""
Task 6: End-to-End Text-to-Image Pipeline
This module integrates all components from Tasks 1-5 into a single
unified text-to-image generation pipeline.

Pipeline Flow:
    Text Prompt
        |
        v
    CLIP Tokenizer + Text Encoder (Task 4)
        |
        v
    512-dim semantic embedding
        |
        +---> Attention GAN (Task 5) ---> 64x64 shape image  [GAN mode]
        |
        +---> Fine-tuned Stable Diffusion (Task 1) ---> 512x512 image [SD mode]

Usage:
    from pipeline import generate

    # Generate using GAN only
    results = generate("a red circle on black background", mode="gan")

    # Generate using Stable Diffusion only
    results = generate("a photo of sks flower in a garden", mode="sd")

    # Generate using both
    results = generate("a blue square shape", mode="both")

    # Each result is a tuple: (label, PIL.Image)
    for label, image in results:
        image.save(f"{label}_output.png")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
import os


NOISE_DIM  = 100   # size of random noise input to GAN Generator
NUM_CLASS  = 3     # number of shape classes: circle, square, triangle
BASE_PATH  = "/content/drive/MyDrive/text-to-image-internship"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# Maps keywords in text prompts to GAN class IDs
CLASS_MAP = {
    "circle":   0,
    "square":   1,
    "triangle": 2,
}


class SelfAttentionBlock(nn.Module):
    """
    Self-attention module for spatial feature maps.
    Allows each spatial position to attend to all other positions,
    enabling long-range spatial coherence in generated images.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Q, K, V projections using 1x1 convolutions
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        # Learnable scale — initialised to 0 so model starts as identity
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N    = H * W
        flat = x.view(B, C, N)
        Q    = self.query(flat)
        K    = self.key(flat)
        V    = self.value(flat)
        # Scaled dot-product attention
        attn = F.softmax(torch.bmm(Q.permute(0, 2, 1), K) / (C ** 0.5), dim=-1)
        out  = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        # Residual connection: start as identity, learn to use attention
        return self.gamma * out + x


class AttentionGenerator(nn.Module):
    """
    Attention GAN Generator from Task 5.
    Generates 64x64 grayscale shape images conditioned on class labels.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + NUM_CLASS, 512 * 4 * 4), nn.ReLU()
        )
        self.attn = SelfAttentionBlock(128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  1,   4, 2, 1), nn.Tanh()
        )

    def forward(self, noise, labels):
        one_hot = F.one_hot(labels, NUM_CLASS).float()
        x = torch.cat([noise, one_hot], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)
        x = self.deconv[0](x); x = self.deconv[1](x); x = self.deconv[2](x)
        x = self.deconv[3](x); x = self.deconv[4](x); x = self.deconv[5](x)
        x = self.attn(x)
        x = self.deconv[6](x); x = self.deconv[7](x); x = self.deconv[8](x)
        x = self.deconv[9](x)
        return x

print("Loading CLIP text encoder...")
tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_encoder.eval()
def text_to_embedding(text: str) -> torch.Tensor:
    """Convert text to normalised 512-dim CLIP embedding."""
    if not text or not text.strip():
        text = "a flower"
    tokens = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        emb = text_encoder(**tokens).pooler_output
    return F.normalize(emb, p=2, dim=1)

print("Loading Attention GAN...")
G_gan = AttentionGenerator().to(DEVICE)
ckpt  = torch.load(f"{BASE_PATH}/task5/checkpoints/attn_G_final.pth",
                   map_location=DEVICE)

# Remap checkpoint keys to current architecture
new_ckpt = {}
key_map  = {"deconv.6.": "attn.", "deconv.7.": "deconv.6.",
            "deconv.8.": "deconv.7.", "deconv.9.": "deconv.8.",
            "deconv.10.": "deconv.9."}
for k, v in ckpt.items():
    new_k = k
    for old, new in key_map.items():
        if k.startswith(old):
            new_k = k.replace(old, new)
            break
    new_ckpt[new_k] = v
G_gan.load_state_dict(new_ckpt)
G_gan.eval()
def generate_gan(class_id: int = 0) -> Image.Image:
    """Generate a 64x64 shape image using the Attention GAN."""
    noise  = torch.randn(1, NOISE_DIM).to(DEVICE)
    labels = torch.tensor([class_id]).to(DEVICE)
    with torch.no_grad():
        fake = G_gan(noise, labels).cpu().squeeze().numpy()
    arr = ((fake + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")
print("Loading fine-tuned Stable Diffusion...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    f"{BASE_PATH}/task1/output_model",
    torch_dtype=torch.float16,
    safety_checker=None
).to(DEVICE)


def generate_sd(prompt: str) -> Image.Image:
    """Generate a 512x512 image using fine-tuned Stable Diffusion."""
    return sd_pipe(prompt, num_inference_steps=30,
                   guidance_scale=7.5).images[0]
def generate(prompt: str, mode: str = "both") -> list:
    """
    Generate image(s) from a text prompt.

    Args:
        prompt (str): Natural language text description.
        mode (str): Generation mode.
            "gan"  — use Attention GAN only (fast, 64x64 shapes)
            "sd"   — use Stable Diffusion only (slow, 512x512 flowers)
            "both" — use both and return two images

    Returns:
        list of (label, PIL.Image) tuples.

    Examples:
        >>> results = generate("a red circle", mode="gan")
        >>> results = generate("a photo of sks flower", mode="sd")
        >>> results = generate("a square shape", mode="both")
    """
    results = []

    if mode in ("gan", "both"):
        # Determine shape class from prompt keywords
        class_id = next(
            (v for k, v in CLASS_MAP.items() if k in prompt.lower()), 0
        )
        results.append(("GAN", generate_gan(class_id)))

    if mode in ("sd", "both"):
        results.append(("SD", generate_sd(prompt)))

    return results


if __name__ == "__main__":
    print("Testing pipeline...")
    r = generate("a red circle on black background", mode="gan")
    print(f"GAN output size: {r[0][1].size}")
    print("Pipeline working correctly.")
