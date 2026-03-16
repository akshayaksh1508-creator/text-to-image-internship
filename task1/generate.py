
import torch
from diffusers import StableDiffusionPipeline
import os

MODEL_PATH = '/content/drive/MyDrive/text-to-image-internship/task1/output_model'
OUTPUT_DIR = '/content/drive/MyDrive/text-to-image-internship/task1/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to('cuda')

prompts = [
    'a photo of sks flower in a garden',
    'a photo of sks flower with morning dew',
    'a photo of sks flower on a white background',
    'a painting of sks flower in watercolour',
    'a photo of sks flower in bright sunlight',
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50).images[0]
    image.save(f'{OUTPUT_DIR}/generated_{i+1:02d}.png')
    print(f'Saved: generated_{i+1:02d}.png')

print('Done! All 5 images generated.')
