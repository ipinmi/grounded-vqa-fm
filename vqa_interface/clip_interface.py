import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_CLIP_batch(dataloader: DataLoader):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    results = {}

    for idx, batch in enumerate(tqdm(dataloader)):
        annot_id, image_paths, questions, answers, labels, detections = batch

        # Assuming one image per batch, open it
        image = Image.open(image_paths[0])

        # Prepare the text inputs (questions paired with answers)
        question_answers = [
            f"<|startoftext|> question: {question} <|endoftext|> <|startoftext|> answer: {answer} <|endoftext|>"
            for question, answer in zip(questions, answers)
        ]

        # Limit to 77 tokens
        question_answers = [qa[:77] for qa in question_answers]

        with torch.no_grad():
            # Create inputs for the model
            inputs = preprocessor(
                text=question_answers,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: [1, num_text_pairs]

            # Softmax to get probabilities
            batch_probs = logits_per_image.softmax(dim=1).squeeze(0).numpy()

            # Convert to one-hot encoded predictions
            predicted_labels = np.argmax(
                batch_probs
            )  # Index of the highest probability
            predicted_probs = np.zeros_like(batch_probs)
            predicted_probs[predicted_labels] = 1

            # image_id = image_paths[0].replace("data/vcr1images/", "") # Map back to image ID
            highest_prob_idx = np.argmax(predicted_probs)

            # Store results
            # "predicted_label": predicted_probs.tolist(),  # label to one-hot sequence
            # "expected_label": labels, label sequence

            results[annot_id[0]] = {
                "question": questions[0],  # Assuming one question per image
                "predicted_answer": answers[highest_prob_idx],  # The predicted answer
                "predicted_index": int(predicted_labels + 1),  # Index to label
                "correct_answer": answers[labels.index(1)],  # The correct answer
                "correct_index": int(labels.index(1) + 1),  # Index to label
            }
            # testing over 10 examples
            if idx == 3001:
                break

    return results


def eval_on_accuracy(results):
    """
    Calculate the accuracy of the model on the VCR dataset.

    Args:
    results (dict): A dictionary of results from the model.

    Returns:
    accuracy (float): The accuracy of the model.
    pred_value_1 (int): The number of predictions that are 1.
    len_results (int): The total number of predictions.
    """
    correct = 0
    # count where predictions equals to 1
    pred_value_1 = 0
    for annot_id, result in results.items():
        if result["predicted_index"] == result["correct_index"]:
            correct += 1

        if result["predicted_index"] == 1:
            pred_value_1 += 1

    if correct == 0:
        return 0.0, pred_value_1, len(results)
    else:
        accuracy = correct / len(results)
        return accuracy, pred_value_1, len(results)
