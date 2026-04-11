"""
Task 5: Self-Attention GAN Architecture and Training
Enhances Task 2 CGAN with self-attention mechanisms (SAGAN-inspired).
Run: python train_attention_gan.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, glob, numpy as np, json

NOISE_DIM = 100
NUM_CLASS  = 3


class SelfAttentionBlock(nn.Module):
    """
    Self-attention module for spatial feature maps.
    Enables long-range spatial dependency modelling.
    Attention(Q,K,V) = softmax(QK^T / sqrt(C)) * V
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels//8, 1)
        self.key   = nn.Conv1d(in_channels, in_channels//8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N    = H * W
        flat = x.view(B, C, N)
        Q    = self.query(flat)
        K    = self.key(flat)
        V    = self.value(flat)
        attn = F.softmax(torch.bmm(Q.permute(0,2,1), K)/(C**0.5), dim=-1)
        out  = torch.bmm(V, attn.permute(0,2,1)).view(B, C, H, W)
        return self.gamma * out + x


class AttentionGenerator(nn.Module):
    """CGAN Generator with self-attention at 16x16 feature map."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM+NUM_CLASS, 512*4*4), nn.ReLU()
        )
        self.attn = SelfAttentionBlock(128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64, 4,2,1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64, 1,  4,2,1), nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, F.one_hot(labels,NUM_CLASS).float()], dim=1)
        x = self.fc(x).view(-1,512,4,4)
        x = self.deconv[0](x); x = self.deconv[1](x); x = self.deconv[2](x)
        x = self.deconv[3](x); x = self.deconv[4](x); x = self.deconv[5](x)
        x = self.attn(x)
        x = self.deconv[6](x); x = self.deconv[7](x); x = self.deconv[8](x)
        return self.deconv[9](x)


class AttentionDiscriminator(nn.Module):
    """CGAN Discriminator with self-attention and label embedding."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASS, 64*64)
        self.conv  = nn.Sequential(
            nn.Conv2d(2,64,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            SelfAttentionBlock(128),
            nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256,1,4,2,1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(16,1), nn.Sigmoid())

    def forward(self, image, labels):
        label_map = self.embed(labels).view(-1,1,64,64)
        return self.fc(self.conv(torch.cat([image, label_map], dim=1)))
