## Task 6: End-to-End Text-to-Image Pipeline

### Objective
Integrate all components from Tasks 1-5 into a unified pipeline accepting
a text prompt and returning a generated image via either the Attention GAN
or fine-tuned Stable Diffusion.

### Pipeline Architecture
- Input: text prompt (string)
- CLIP (Task 4): tokenises and encodes text to 512-dim embedding
- Attention GAN (Task 5): uses class label for 64x64 shape generation
- Fine-tuned SD (Task 1): uses prompt for 512x512 flower generation

### Gradio Interface
Interactive web UI with text input and mode dropdown (GAN / Stable Diffusion).
Tested with 10 prompts. Public link generated via share=True.

### Results
GAN mode generates shapes in under 1 second.
SD mode generates photorealistic flowers in ~10 seconds.
5 test output images saved to task6/outputs/.
