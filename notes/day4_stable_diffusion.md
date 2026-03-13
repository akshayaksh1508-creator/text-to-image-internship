
# Day 4 — Stable Diffusion Architecture
## 3 Components
1. CLIP — text → 512-dim embedding
2. VAE  — image ↔ latent space (8x smaller)
3. UNet — denoises latents guided by text embeddings
## Diffusion Process
- Forward  : image → add noise → pure noise
- Reverse  : pure noise → remove noise → new image
- UNet runs 20-50 steps per generation
# Fine-tuning Stable Diffusion

## DreamBooth (Used in Task 1)
- Updates UNet weights directly
- Instance prompt: 'a photo of sks [subject]'
- ~800 steps, saves full model (~4GB)
- Best quality for domain-specific images

## Textual Inversion
- Does NOT change model weights
- Learns a new text token [V]
- Tiny output (~50KB)
- Better for style transfer

## Decision: USE DREAMBOOTH for Task 1
