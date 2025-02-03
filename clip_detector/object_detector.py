import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_patches(image_path, patch_size):
    original_image = Image.open(image_path)

    transformer = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # converting the PIL image to tensor (C, H, W)
    transformed_image = transformer(original_image)

    # Add batch dimension and shift color channels behind the height and width
    transformed_image = transformed_image.unfold(0, 3, 3)

    # split into equal patches
    patches = transformed_image.unfold(1, patch_size, patch_size)
    patches = patches.unfold(2, patch_size, patch_size)

    # Number of patches in X and Y direction
    X, Y = patches.shape[1], patches.shape[2]

    return patches, original_image, transformed_image


def generate_prompts(object_categories):
    obj_prompts = []
    for obj in object_categories:
        obj_prompts.append(f"a photo of a {obj}")
    return obj_prompts


def CLIPDetect(window, stride, patches, patch_size, current_prompt):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scores = torch.zeros(patches.shape[1], patches.shape[2])
    runs = torch.ones(patches.shape[1], patches.shape[2])

    for Y in range(0, patches.shape[1] - window + 1, stride):
        for X in range(0, patches.shape[2] - window + 1, stride):
            # Create a big patch from the current window
            window_patch = torch.zeros(patch_size * window, patch_size * window, 3)
            patch_batch = patches[0, Y : Y + window, X : X + window]

            for y in range(window):
                for x in range(window):
                    window_patch[
                        y * patch_size : (y + 1) * patch_size,
                        x * patch_size : (x + 1) * patch_size,
                        :,
                    ] = patch_batch[y, x].permute(1, 2, 0)

            inputs = processor(
                images=window_patch,
                return_tensors="pt",
                text=current_prompt,
                padding=True,
            ).to(device)

            outputs = model(**inputs)
            score = outputs.logits_per_image.item()  # Shape: [1, num_text_pairs]

            # sum of all scores on each patch within the current window
            scores[Y : Y + window, X : X + window] += score
            # number of runs on each patch within the current window
            runs[Y : Y + window, X : X + window] += 1

    # average scores over the number of runs
    scores /= runs

    # clip the interval edges to avoid outliers
    for _ in range(3):
        scores = np.clip(scores - scores.mean(), 0, np.inf)

    # normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    return scores


def extract_bounding_boxes(scores, patch_size, decision_threshold):
    # Create a mask for scores above the decision threshold
    detection = scores > decision_threshold

    # Find the bounding boxes
    bounding_boxes = []

    y_min, y_max = (
        np.nonzero(detection)[:, 0].min().item(),
        np.nonzero(detection)[:, 0].max().item() + 1,
    )

    x_min, x_max = (
        np.nonzero(detection)[:, 1].min().item(),
        np.nonzero(detection)[:, 1].max().item() + 1,
    )
    y_min *= patch_size
    y_max *= patch_size
    x_min *= patch_size
    x_max *= patch_size

    # height and width of the bounding box
    obj_height = y_max - y_min
    obj_width = x_max - x_min

    bounding_boxes.append(
        {
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
            "obj_height": obj_height,
            "obj_width": obj_width,
        }
    )

    return bounding_boxes


def run_detection(
    image_path, object_cats, patch_size, window, stride, decision_threshold
):

    # Get image patches
    img_patches, original_image, transformed_image = create_patches(
        image_path, patch_size
    )

    obj_prompts = generate_prompts(object_cats)

    # image = np.moveaxis(transformed_image.data.numpy(), 0, -1)

    # original_image.show()

    # Run the detector for each object

    boxes_results = {}

    for i, current_prompt in enumerate(obj_prompts):

        scores = CLIPDetect(window, stride, img_patches, patch_size, current_prompt)

        bounding_box = extract_bounding_boxes(scores, patch_size, decision_threshold)

        # print(f"Prompt: {current_prompt}, Bounding Box: {bounding_box}")

        object_name = current_prompt.split(" ")[-1]

        boxes_results[object_name] = bounding_box

    return boxes_results


def draw_box(bounding_results, original_image, i):

    x_min, y_min, x_max, y_max = (
        bounding_results["x_min"],
        bounding_results["y_min"],
        bounding_results["x_max"],
        bounding_results["y_max"],
    )

    obj_height = bounding_results["obj_height"]
    obj_width = bounding_results["obj_width"]

    cropped_image = original_image.crop(
        (x_min, y_min, x_max + obj_width, y_max + obj_height)
    )

    cropped_image.save(f"cropped_patch_{i}.png")  # Save as a file

    cropped_image.show()


def calculate_iou(ground_truth, predicted_box):
    """
    Calculate the Intersection over Union (IoU) between a ground truth box and a predicted box.

    Args:
    ground_truth (list): Bounding box in the format [x1, y1, x2, y2].
    predicted_box (dict): Bounding box with keys {'x_min', 'y_min', 'x_max', 'y_max'}.

    Returns:
    float: IoU value (0 if no overlap).
    """
    # Calculate intersection dimensions
    intersection_width = max(
        0,
        min(ground_truth[2], predicted_box["x_max"])
        - max(ground_truth[0], predicted_box["x_min"]),
    )
    intersection_height = max(
        0,
        min(ground_truth[3], predicted_box["y_max"])
        - max(ground_truth[1], predicted_box["y_min"]),
    )

    # If no intersection, IoU is 0
    if intersection_width == 0 or intersection_height == 0:
        return 0.0

    # Compute areas
    intersection_area = intersection_width * intersection_height
    ground_truth_area = (ground_truth[2] - ground_truth[0]) * (
        ground_truth[3] - ground_truth[1]
    )
    predicted_area = (predicted_box["x_max"] - predicted_box["x_min"]) * (
        predicted_box["y_max"] - predicted_box["y_min"]
    )

    # Compute IoU
    union_area = ground_truth_area + predicted_area - intersection_area
    iou = intersection_area / union_area

    return iou
