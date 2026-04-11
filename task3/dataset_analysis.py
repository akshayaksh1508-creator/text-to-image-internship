"""
Task 3: Oxford-102 Flowers Dataset Analysis
Loads dataset, analyses class distribution, resolution stats,
and produces visualisation outputs.
Run: python dataset_analysis.py
"""
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np, os, random

os.makedirs('outputs', exist_ok=True)

# Load dataset
print('Loading Oxford-102 Flowers dataset...')
dataset = load_dataset('huggan/flowers-102-categories', split='train')
print(f'Total images: {len(dataset)}')

# Class distribution
images_per_class = len(dataset) // 102
labels = [min(i // images_per_class, 101) for i in range(len(dataset))]
class_counts = Counter(labels)

print(f'Total classes  : {len(class_counts)}')
print(f'Avg per class  : {len(dataset)/len(class_counts):.1f}')
print(f'Max per class  : {max(class_counts.values())}')
print(f'Min per class  : {min(class_counts.values())}')

# Bar chart
sorted_counts = sorted(class_counts.items())
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar([str(k) for k,v in sorted_counts],
       [v for k,v in sorted_counts], color='#a5b4fc')
ax.set_title('Oxford-102 — Images per Class')
ax.set_xlabel('Class ID'); ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/class_distribution.png', dpi=80)
plt.close()
print('Class distribution chart saved.')

# Sample image grid
indices = random.sample(range(len(dataset)), 20)
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for i, idx in enumerate(indices):
    axes[i//5][i%5].imshow(dataset[idx]['image'])
    axes[i//5][i%5].set_title(f'Class {labels[idx]}', fontsize=8)
    axes[i//5][i%5].axis('off')
plt.tight_layout()
plt.savefig('outputs/sample_images.png', dpi=80)
plt.close()
print('Sample grid saved.')

# Resolution stats
widths, heights = [], []
for i in random.sample(range(len(dataset)), 200):
    img = dataset[i]['image']
    widths.append(img.size[0])
    heights.append(img.size[1])
print(f'Width  avg: {sum(widths)/len(widths):.0f}')
print(f'Height avg: {sum(heights)/len(heights):.0f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths, bins=20, color='#a5b4fc')
axes[0].set_title('Width Distribution')
axes[1].hist(heights, bins=20, color='#6ee7b7')
axes[1].set_title('Height Distribution')
plt.tight_layout()
plt.savefig('outputs/resolution_histogram.png', dpi=80)
plt.close()
print('Resolution histogram saved.')
print('Task 3 analysis complete.')
