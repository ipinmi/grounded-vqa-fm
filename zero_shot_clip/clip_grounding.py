# Code adapted from: https://github.com/hila-chefer/Transformer-MM-Explainability

import torch
import clip
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


from captum.attr import visualization
import cv2
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocessor = clip.load("ViT-B/32", device=device, jit=False)

# Transformer attention layers to consider
# for text encoder
text_layer = -1  # Last layer

# for vision encoder
vision_layer = -1  # Last layer


def interpret(
    image,
    texts,
    model,
    device,
    vision_start_layer=vision_layer,
    text_start_layer=text_layer,
):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros(
        (logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32
    )
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values()
    )

    if vision_start_layer == -1:
        # calculate index of last layer
        vision_start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(
        num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
    ).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < vision_start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[
            0
        ].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if text_start_layer == -1:
        # calculate index of last layer
        text_start_layer = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(
        num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype
    ).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < text_start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[
            0
        ].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance


def show_image_with_bounding_boxes(image_relevance, image, orig_image, threshold=0.5):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode="bilinear"
    )
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    binary_mask = (image_relevance > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    for x, y, w, h in bounding_boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis("off")
    axs[1].imshow(vis)
    axs[1].axis("off")
    plt.show()


def show_heatmap_on_text(text, text_encoding, R_text):
    _tokenizer = _Tokenizer()
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    print(text_scores)
    text_tokens = _tokenizer.encode(text)
    text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [
        visualization.VisualizationDataRecord(
            text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1
        )
    ]
    visualization.visualize_text(vis_data_records)


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def attention_viz():
    pass
