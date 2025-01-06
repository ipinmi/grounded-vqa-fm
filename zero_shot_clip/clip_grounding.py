# Code adapted from: https://github.com/hila-chefer/Transformer-MM-Explainability

import torch
import CLIP.clip as clip  # modified CLIP model from the repository
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization


#### START OF CODE ADAPTATION ####


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


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load("ViT-B/32", device=device, jit=False)

# Transformer attention layers to consider
# for text encoder
text_layer = -1  # Last layer

# for vision encoder
vision_layer = -1  # Last layer


def interpret(
    image,
    texts,
    model,
    vision_start_layer=vision_layer,
    text_start_layer=text_layer,
):
    """
    Interpret the model by calculating the relevance scores for the image and text inputs.

    Args:
    image (torch.Tensor): The image tensor.
    texts (torch.Tensor): The text tensor.
    model (torch.nn.Module): The CLIP model.
    vision_start_layer (int): The starting layer for the vision encoder.
    text_start_layer (int): The starting layer for the text encoder.

    Returns:
    text_relevance (torch.Tensor): The relevance scores for the text input.
    image_relevance (torch.Tensor): The relevance scores for the image input.
    """
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

    # return bounding_boxes


#### END OF CODE ADAPTATION ####


def get_prediction(image, text, model, choices=None):
    """
    Get the predicted label for the image-text pair.

    Args:
    image: torch.Tensor. The image tensor.
    text: torch.Tensor. The text tensor.
    model: torch.nn.Module. The CLIP model.

    Returns:
    predicted_label: int. The predicted label.

    """
    if choices is None:
        logits_per_image, logits_per_text = model(image, text)
        print(
            color.BOLD
            + color.BLUE
            + color.UNDERLINE
            + f"CLIP similarity score: {logits_per_image.item()}"
            + color.END
        )

        # Softmax to get probabilities
        batch_probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()

        # Convert to one-hot encoded predictions
        predicted_label = np.argmax(batch_probs)  # Index of the highest probability

    else:
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        answers_features = model.encode_text(choices).to(device)  # shape: (4, 512)
        answers_features /= answers_features.norm(dim=-1, keepdim=True)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        answers_features /= answers_features.norm(dim=-1, keepdim=True)

        joint_features = text_features + image_features

        # Compute the similarity between the joint and answers features
        similarity_array = joint_features @ answers_features.T

        _, predicted_label = similarity_array.max(dim=-1)

    return predicted_label


def attention_visualizaton(
    dataloader,
    model,
    mode: str = "answer",
    num_samples: int = 10,
    num_objects: list = [3, 5, 7, 10],
):
    """
    Visualize the attention maps for the text and image inputs using CLIP model.

    Args:
    dataloader: DataLoader object containing the data.
    model: CLIP model.
    predicted_label: int. the answer label predicted by the model.
    mode: str, default "answer". Decides whether to include the answer in the text input or not.
        Options: "answer" or "no answer".
    num_samples: int, default 10. Number of samples to visualize.
    num_objects: list, default [3, 5, 7, 10]. Number of relevant objects in the image for calculating the ROI accuracy.

    Returns:

    """
    bounding_results = {}  # store bounding box values

    for idx, batch in enumerate(tqdm(dataloader)):
        annot_id, image_paths, questions, choices, labels, detections = batch

        # Assuming one image per batch, load and preprocess it
        img_path = image_paths[0]
        image = preprocessor(Image.open(img_path)).unsqueeze(0).to(device)

        # Prepare the text inputs (question and only the correct and predicted choices) and image inputs
        if mode == "answer":
            question_answers = [
                f"{question} {answer}" for question, answer in zip(questions, choices)
            ]  # shape: (num of choices, 77)

            # Limit to 77 tokens to fit in the model
            question_answers = [qa[:77] for qa in question_answers]

            text_input = clip.tokenize(question_answers).to(
                device
            )  # shape: (num_choices, 77)

            predicted_label = get_prediction(image, text_input, model, None)

        elif mode == "no_answer":
            question_tokens = clip.tokenize(questions[0]).to(device)  # shape: (1, 77)
            choices_tokens = clip.tokenize(choices).to(
                device
            )  # shape: (num of choices, 77)

            text_input = question_tokens

            predicted_label = get_prediction(image, text_input, model, choices_tokens)

        # Select the predicted and correct choices for visualization
        if predicted_label == labels.index(1):
            text_input = text_input[predicted_label]
        elif predicted_label < labels.index(1):
            text_input = text_input[[predicted_label, labels.index(1)]]
        else:
            text_input = text_input[[labels.index(1), predicted_label]]

        # Interpret the model and visualize the attention maps
        R_text, R_image = interpret(model=model, image=image, texts=text_input)
        batch_size = text_input.shape[0]
        for i in range(batch_size):
            show_heatmap_on_text(text_input[i], text_input[i], R_text[i])
            bounding_box = show_image_with_bounding_boxes(
                R_image[i], image, orig_image=Image.open(img_path)
            )
            plt.show()

            # save bounding box values in a json file
            bounding_results[annot_id] = {
                "question": questions[0],
                "correct_answer": choices[labels.index(1)],
                "image_path": img_path,
                "orig_bounding_boxes": detections["boxes"],
                "pred_bounding_boxes": bounding_box,
            }

        if idx == num_samples:
            break
        # NOTE: create viz folder and save bounding box images, pass similarity score to function
