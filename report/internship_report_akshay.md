# Internship Report — Text-to-Image AI Engineering
## Akshay H | Elevance Skills | 84 Days

---

## Section 1: Introduction

This report documents a 3-month internship at Elevance Skills focused on
Text-to-Image AI engineering. The programme covered six core tasks progressing
from foundational GAN architectures through to a complete deployed pipeline.
All work was implemented in Python using PyTorch and Hugging Face libraries
on Google Colab, with all code version-controlled on GitHub throughout.

The six tasks completed are:
1. Fine-tuning Stable Diffusion v1.5 using DreamBooth on 25 flower images
2. Building a Conditional GAN for geometric shape generation from class labels
3. Loading and analysing the Oxford-102 Flowers public dataset
4. Implementing a reusable CLIP-based text embedding pipeline
5. Enhancing the CGAN with self-attention mechanisms inspired by SAGAN
6. Assembling all components into an end-to-end pipeline with Gradio UI

---

## Section 2: Background

Text-to-image generation is a field of AI where models produce realistic
images from natural language descriptions. GANs use a Generator-Discriminator
adversarial training loop. Conditional GANs extend this with class labels.
Stable Diffusion uses forward diffusion (adding noise) and reverse denoising
guided by CLIP text embeddings in a compressed VAE latent space. DreamBooth
fine-tunes the UNet using a small dataset with a rare token identifier (sks).
Self-Attention GANs (SAGAN) enable long-range spatial coherence in outputs.

---

## Section 3: Learning Objectives

Objectives set at the start:
1. Understand and implement GANs and CGANs from first principles in PyTorch
2. Understand Stable Diffusion architecture — CLIP, VAE, and UNet
3. Fine-tune Stable Diffusion using DreamBooth on custom images
4. Build a CGAN for controlled class-specific image generation
5. Develop proficiency with Hugging Face Transformers and Diffusers
6. Implement self-attention in a GAN architecture
7. Build an end-to-end text-to-image pipeline with an interactive UI

All 7 objectives were fully achieved. The most challenging was GPU memory
management during DreamBooth training, requiring fp16 mixed precision,
gradient checkpointing, and 8-bit Adam to fit within Colab T4 VRAM limits.

---

## Section 4: Activities and Tasks

### Task 1: Fine-tuning Stable Diffusion with DreamBooth
Dataset: 25 Oxford-102 flower images at 512x512 pixels. Instance prompt:
a photo of sks flower. Training: 800 steps, lr 2e-6, fp16, gradient
checkpointing, 8-bit Adam, T4 GPU, approximately 35 minutes.
Result: Fine-tuned model generates domain-specific flowers matching the
training dataset. Comparison grid shows clear improvement over base model.

### Task 2: Conditional GAN for Shape Generation
Synthetic dataset: 3000 images — 1000 circles, 1000 squares, 1000 triangles.
Generator: noise(100) + one_hot(3) concatenated, Linear layers with BatchNorm,
Tanh output, 64x64 image. Discriminator: image + label, LeakyReLU layers,
Sigmoid output. Trained 100 epochs with BCE loss and Adam optimiser.

### Task 3: Oxford-102 Flowers Dataset Analysis
Loaded 8189 images from huggan/flowers-102-categories. Analysed class
distribution across 102 categories. Produced class distribution bar chart,
20-image sample grid, and resolution histograms. Average resolution ~500x400.

### Task 4: Text Preprocessing Pipeline
Built text_to_embedding() and texts_to_embeddings() using CLIPTokenizer and
CLIPTextModel. Outputs L2-normalised 512-dim embeddings. Validated via cosine
similarity. Edge cases handled gracefully. Saved as embedding_pipeline.py.

### Task 5: Self-Attention GAN
Implemented SelfAttentionBlock with Conv1d Q/K/V projections and learnable
gamma. Integrated into convolutional Generator after 16x16 layer and into
Discriminator. Trained 100 epochs on Task 2 shape dataset. Produced more
geometrically regular shapes compared to baseline CGAN.

### Task 6: End-to-End Pipeline
Integrated CLIP (Task 4) + Attention GAN (Task 5) + fine-tuned SD (Task 1)
into pipeline.py with generate(prompt, mode) function. Built Gradio web
interface with text input and mode selector. Tested with multiple prompts.
GAN mode under 1 second. SD mode approximately 10 seconds per image.

---

## Section 5: Skills and Competencies

Technical: PyTorch (custom modules, training loops, GPU management),
Hugging Face (Diffusers, Transformers, Datasets), CLIP embeddings,
DreamBooth fine-tuning, self-attention, Gradio UI, matplotlib visualisation.
Tools: Google Colab, GitHub daily commits, Google Drive, PIL, NumPy.
Soft skills: Self-directed research, independent debugging, documentation,
time management across 12 weeks.

---

## Section 6: Feedback and Evidence

GitHub: https://github.com/akshayaksh1508-creator/text-to-image-internship

Key outputs:
- task1/outputs/comparison_grid.png — base vs fine-tuned SD comparison
- task2/outputs/loss_curve.png — CGAN training loss over 100 epochs
- task2/outputs/circle_grid.png, square_grid.png, triangle_grid.png
- task3/outputs/class_distribution.png, sample_images.png
- task5/outputs/attn_loss_curve.png — Attention GAN training loss
- task6/outputs/ — 5 end-to-end pipeline test outputs

Daily form submitted every working day across all 84 days.

---

## Section 7: Challenges and Solutions

1. CUDA Out of Memory (DreamBooth): Fixed with fp16 + gradient checkpointing
   + 8-bit Adam. Peak memory reduced from 15GB to 10GB.

2. NameError in Colab cells: Fixed by running all cells top-to-bottom at
   start of every session. All imports repeated in each major cell.

3. Dataset not found (huggan/flowers-102): Fixed by using correct name
   huggan/flowers-102-categories from Hugging Face Hub.

4. DreamBooth version mismatch: Fixed by installing diffusers from source
   using pip install git+https://github.com/huggingface/diffusers

5. Tensor shape mismatch (Attention GAN): Fixed by adding explicit
   .view(B, C, H, W) after attention matrix multiplication output.

---

## Section 8: Outcomes and Impact

Deliverables:
1. Fine-tuned DreamBooth SD model generating domain-specific flower images
2. Trained CGAN producing circles, squares, triangles from class labels
3. Complete Oxford-102 EDA with class distribution and resolution analysis
4. Reusable CLIP embedding pipeline used directly in Task 6
5. Self-attention GAN with improved geometric coherence over baseline
6. End-to-end interactive text-to-image system via Gradio web interface

Real-world applicability: DreamBooth pipeline mirrors commercial personalised
image generation tools. CLIP embedding pipeline is standard in production
multimodal AI systems. Portfolio provides concrete AI engineering evidence.

---

## Section 9: Conclusion

This 3-month internship provided comprehensive hands-on experience building
text-to-image AI systems from first principles. Beginning with GAN theory and
ending with a fully deployed interactive pipeline, the programme successfully
bridged theoretical knowledge and practical engineering skill.

Each task built progressively on the previous ones — CGAN theory became the
Task 2 implementation, Stable Diffusion architecture study became the Task 1
fine-tuning project, CLIP embeddings from Task 4 became the Task 6 pipeline
core. This integration gave a genuinely holistic understanding of modern
text-to-image systems.

The most valuable learning was entirely practical — debugging GPU memory
errors, fixing tensor shape mismatches, resolving library version conflicts,
and recovering from Colab session resets. These are skills that cannot be
acquired from coursework alone.

Future directions: training the attention GAN on real natural images,
exploring LoRA as a lighter DreamBooth alternative, and deploying the
Gradio app on Hugging Face Spaces for public access.
