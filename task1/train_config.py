"""
Task 1: DreamBooth Fine-tuning Configuration

This file contains all training arguments for fine-tuning
Stable Diffusion v1.5 using the DreamBooth technique on a
custom Oxford-102 flowers dataset.

Usage:
    Run train_dreambooth.py with these arguments to fine-tune the model.
    The fine-tuned model is saved to output_dir for later inference.
"""

# Base model to fine-tune — Stable Diffusion v1.5 from Hugging Face
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"

# Path to your 25 training images (512x512 JPG format)
INSTANCE_DATA_DIR = "/content/drive/MyDrive/text-to-image-internship/task1/dataset/images"

# DreamBooth identifier token — rare word that represents YOUR subject
# After training, "sks flower" generates YOUR specific flowers
INSTANCE_PROMPT = "a photo of sks flower"

# Where to save the fine-tuned model weights
OUTPUT_DIR = "/content/drive/MyDrive/text-to-image-internship/task1/output_model"

# Image resolution — must match your dataset images
RESOLUTION = 512

# Number of images processed per training step
# Keep at 1 for Colab T4 to avoid CUDA out-of-memory errors
TRAIN_BATCH_SIZE = 1

# Total number of gradient update steps
# 800 steps is sufficient for 25 images with DreamBooth
MAX_TRAIN_STEPS = 800

# Learning rate — how fast the model updates weights
# 2e-6 is recommended for DreamBooth to avoid overfitting
LEARNING_RATE = 2e-6

# Use 16-bit floating point — halves GPU memory usage
MIXED_PRECISION = "fp16"

# Recompute activations during backward pass instead of storing them
# Critical for fitting DreamBooth in Colab T4 VRAM (15GB)
GRADIENT_CHECKPOINTING = True

# Memory-efficient Adam optimiser using 8-bit quantisation
# Reduces optimiser state memory by ~75%
USE_8BIT_ADAM = True

# Save a checkpoint every 400 steps so training can resume if interrupted
CHECKPOINTING_STEPS = 400
