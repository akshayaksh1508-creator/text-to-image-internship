# Day 2 Notes — GAN Study

## What is a GAN?
- GAN = Generative Adversarial Network
- Two networks: Generator (G) vs Discriminator (D)
- G creates fake images from random noise
- D tries to tell real from fake

## Generator
- Input: random noise vector (size 100)
- Output: fake image (784 = 28x28)
- Goal: fool D into thinking output is real

## Discriminator
- Input: image (real or fake)
- Output: probability 0 (fake) to 1 (real)
- Goal: correctly label real=1, fake=0

## Loss Functions
- Both use BCE (Binary Cross Entropy)
- D loss = loss on real + loss on fake
- G loss = how well G fooled D
- Ideal: both losses settle near 0.6-0.7

## Common Problems
- Mode collapse: G generates same image every time
- Vanishing gradient: D too strong, G cant learn
