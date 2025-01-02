import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from captum.attr import visualization
import cv2
import matplotlib.pyplot as plt

# Transformer layers to consider
# for text encoder
text_layer = -1  # Last layer

# for vision encoder
vision_layer = -1  # Last layer


def attention_viz():
    pass
