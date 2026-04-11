"""
Task 2: Synthetic Shape Dataset Generator
Generates 3000 images: 1000 circles, 1000 squares, 1000 triangles
Run: python generate_dataset.py
"""
from PIL import Image, ImageDraw
import os, random

IMG_SIZE = 64
N        = 1000
BASE     = 'dataset'

for cls in ['circle', 'square', 'triangle']:
    os.makedirs(f'{BASE}/{cls}', exist_ok=True)

def make_circle():
    img  = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(img)
    r  = random.randint(15, 25)
    cx = random.randint(r+5, IMG_SIZE-r-5)
    cy = random.randint(r+5, IMG_SIZE-r-5)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)
    return img

def make_square():
    img  = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(img)
    s  = random.randint(20, 35)
    x  = random.randint(5, IMG_SIZE-s-5)
    y  = random.randint(5, IMG_SIZE-s-5)
    draw.rectangle([x, y, x+s, y+s], fill=255)
    return img

def make_triangle():
    img  = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = random.randint(20, 44)
    cy = random.randint(20, 44)
    r  = random.randint(15, 22)
    draw.polygon([(cx, cy-r),(cx-r, cy+r),(cx+r, cy+r)], fill=255)
    return img

print(f'Generating {N*3} images...')
for i in range(N):
    make_circle().save(f'{BASE}/circle/circle_{i:04d}.png')
    make_square().save(f'{BASE}/square/square_{i:04d}.png')
    make_triangle().save(f'{BASE}/triangle/triangle_{i:04d}.png')
print('Done.')
