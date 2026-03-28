
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np

# text_to_embedding(), generate_gan(), generate_sd(), generate()
# Full implementation in Session 12 notebook
