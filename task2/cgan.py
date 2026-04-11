"""
Task 2: Conditional GAN (CGAN) for Geometric Shape Generation

This module defines the Generator and Discriminator architectures
for a Conditional GAN that generates 64x64 grayscale images of
three geometric shapes: circle (label=0), square (label=1),
triangle (label=2).

Architecture Overview:
    Generator  : noise(100) + one_hot(3) -> Linear layers -> 64x64 image
    Discriminator: image(4096) + one_hot(3) -> Linear layers -> real/fake

Key Design Choices:
    - One-hot encoding converts integer labels to vectors for concatenation
    - BatchNorm1d in Generator stabilises training and prevents mode collapse
    - LeakyReLU(0.2) in Discriminator prevents dead neurons
    - Tanh output in Generator maps pixel values to [-1, 1] range
    - Label smoothing (0.9 instead of 1.0) prevents Discriminator overconfidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimension of the random noise input vector
NOISE_DIM = 100

# Number of shape classes: circle=0, square=1, triangle=2
NUM_CLASS = 3

# Image dimensions: 64x64 pixels flattened to a single vector
IMG_DIM = 64 * 64  # 4096


class Generator(nn.Module):
    """
    CGAN Generator Network.

    Takes a concatenation of random noise and a one-hot class label,
    and produces a 64x64 grayscale image of the specified shape class.

    Input:
        noise  (Tensor): Random noise vector of shape (batch, 100)
        labels (Tensor): Integer class labels of shape (batch,)

    Output:
        Tensor of shape (batch, 1, 64, 64) with values in [-1, 1]
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: expand from 103 (noise+label) to 256 features
            nn.Linear(NOISE_DIM + NUM_CLASS, 256),
            nn.BatchNorm1d(256),  # normalise activations for stable training
            nn.ReLU(),

            # Layer 2: expand from 256 to 512 features
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # Layer 3: expand from 512 to 1024 features
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # Output layer: map 1024 features to 4096 pixels (64x64)
            nn.Linear(1024, IMG_DIM),
            nn.Tanh()  # output in [-1, 1] to match normalised images
        )

    def forward(self, noise, labels):
        # Convert integer labels to one-hot vectors: e.g. 0 -> [1, 0, 0]
        one_hot = F.one_hot(labels, num_classes=NUM_CLASS).float()

        # Concatenate noise and label: [noise(100), one_hot(3)] -> (103,)
        x = torch.cat([noise, one_hot], dim=1)

        # Pass through layers and reshape to image format
        return self.model(x).view(-1, 1, 64, 64)


class Discriminator(nn.Module):
    """
    CGAN Discriminator Network.

    Takes a concatenation of an image and a one-hot class label,
    and outputs the probability that the image is real (not generated).

    Input:
        image  (Tensor): Image tensor of shape (batch, 1, 64, 64)
        labels (Tensor): Integer class labels of shape (batch,)

    Output:
        Tensor of shape (batch, 1) with values in [0, 1]
        where 1 = real and 0 = fake
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: process 4099 (image+label) features -> 512
            nn.Linear(IMG_DIM + NUM_CLASS, 512),
            nn.LeakyReLU(0.2),  # allows small gradient for negative values

            # Layer 2: compress 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            # Output layer: single probability value
            nn.Linear(256, 1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def forward(self, image, labels):
        # Flatten image from (batch, 1, 64, 64) to (batch, 4096)
        flat_image = image.view(image.size(0), -1)

        # Convert labels to one-hot vectors
        one_hot = F.one_hot(labels, num_classes=NUM_CLASS).float()

        # Concatenate image and label: [image(4096), one_hot(3)] -> (4099,)
        x = torch.cat([flat_image, one_hot], dim=1)

        return self.model(x)
