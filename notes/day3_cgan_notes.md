
# Day 3 Notes — Conditional GAN (CGAN)

## What is a CGAN?
- Same as GAN but G and D receive an extra label input
- Label tells G what class to generate (circle, square, triangle)
- Label tells D what class to expect

## One-Hot Encoding
- circle   = [1, 0, 0]
- square   = [0, 1, 0]
- triangle = [0, 0, 1]
- Concatenated with noise or image before feeding to network

## CGAN Generator
- Input : noise(100) + label(3) = size 103
- Output: fake image (784 = 28x28)

## CGAN Discriminator
- Input : image(784) + label(3) = size 787
- Output: probability 0 (fake) to 1 (real)

## This is exactly Task 2!
- Task 2 = build CGAN for circles, squares, triangles
- Architecture today = blueprint for Task 2 in Week 4
