import torch
import clip
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocessor = clip.load("ViT-B/32", device=device)


def run_CLIP_on_VCR(dataloader: DataLoader):
    """
    Run the CLIP model on a batch of VCR data.

    Args:
    dataloader (DataLoader): A DataLoader object containing the VQA data.

    Returns:
    results (dict): A nested dictionary of results from the model containing the following keys:
        - annot_id (str): The annotation ID.
        - question (str): The question.
        - predicted_answer (str): The predicted answer.
        - predicted_index (int): The index of the predicted answer.
        - correct_answer (str): The correct answer.
        - correct_index (int): The index of the correct answer.
    """

    results = {}
    correct = 0

    for idx, batch in enumerate(tqdm(dataloader)):

        annot_id, image_paths, questions, answers, labels, detections = batch

        # Assuming one image per batch, open and preprocess it
        image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

        # Prepare the text inputs (questions paired with answers)
        question_answers = [
            f"{question} {answer}" for question, answer in zip(questions, answers)
        ]
        # Limit to 77 tokens to fit in the model
        question_answers = [qa[:77] for qa in question_answers]

        # Prepare the image inputs
        # images = [image for _ in range(len(question_answers))]  # Repeat the image
        # images = image.repeat(len(question_answers), 1, 1, 1)

        text_input = clip.tokenize(question_answers).to(device)

        with torch.no_grad():
            # Create inputs for the model
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_input)

            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute the similarity between the text and image features
            logits_per_image, logits_per_text = model(
                image, text_input
            )  # Shape: [num_images, num_text_pairs]

            # Softmax to get probabilities
            batch_probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()

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

            # Check if the predicted label is correct
            if (predicted_labels + 1) == labels.index(1) + 1:
                correct += 1

            results[annot_id[0]] = {
                "question": questions[0],  # Assuming one question per image
                "predicted_answer": answers[highest_prob_idx],  # The predicted answer
                "predicted_index": int(predicted_labels + 1),  # Index to label
                "correct_answer": answers[labels.index(1)],  # The correct answer
                "correct_index": int(labels.index(1) + 1),  # Index to label
            }
    # Calculate accuracy
    if correct == 0:
        accuracy = 0.0
    else:
        accuracy = correct / len(results)

    print(f"Accuracy over the VCR data with access to answers: {accuracy}")
    return results


def run_CLIP_on_VQA(dataloader: DataLoader):
    """
    Run the CLIP model on a batch of VQA V2 data.

    Args:
    dataloader (DataLoader): A DataLoader object containing the VQA data.

    Returns:
    results (list): A list of dictionaries containing the following keys:
        - annot_id (int): The annotation ID.
        - question (str): The question.
        - predicted_label (list): The predicted label = 1 if the probability is greater than 0.5. Otherwise, 0.
    """

    results = []
    correct = 0

    for idx, batch in enumerate(tqdm(dataloader)):
        annot_ids = batch["annot_id"].detach().numpy()
        questions = batch["question"]
        answers = batch["answer"]
        image_paths = batch["image_path"]
        answer_types = batch["answer_type"]

        # Prepare the text inputs (each question paired with an answer)
        questions_answers = [
            f"{question} {answer}" for question, answer in zip(questions, answers)
        ]
        text_input = clip.tokenize(questions_answers).to(device)

        images = [Image.open(image_path) for image_path in image_paths]

        image_features = torch.stack([preprocessor(i) for i in images]).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(
                image=image_features, text=text_input
            )

            # Softmax to get probabilities
            batch_probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()

            highest_prob = np.max(
                batch_probs, axis=1
            )  # highest probability for each image

            # set threshold for binary classification
            predicted_labels = (highest_prob >= 0.8).astype(
                int
            )  # NOTE: adjust threshold here
            predicted_labels = predicted_labels.tolist()

        for i, annot_id in enumerate(annot_ids):
            if predicted_labels[i] == 1:
                correct += 1

            results.append(
                {
                    "annot_id": int(annot_id),
                    "image_path": image_paths[i],
                    "question": questions[i],
                    "answer": answers[i],
                    "predicted_label": predicted_labels[i],
                    "answer_type": answer_types[i],
                }
            )
    if correct == 0:
        accuracy = 0.0
    else:
        accuracy = correct / len(results)

    print(f"Accuracy over the VQA V2 data with access to answers: {accuracy}")

    return results


def eval_on_accuracy(results, dataset):
    """
    Calculate the accuracy of the model on the VCR dataset.

    Args:
    results (dict): A dictionary of results from the model.
    dataset (str): The dataset used for evaluation.
        - "VCR" for VCR dataset
        - "VQA" for VQA V2 dataset

    Returns:
    accuracy (float): The accuracy of the model.
    pred_value_1 (int): The number of predictions that are 1.
    len_results (int): The total number of predictions.
    """
    correct = 0
    # count where predictions equals to 1
    pred_value_1 = 0

    if dataset == "vcr":
        for annot_id, result in results.items():
            if result["predicted_index"] == result["correct_index"]:
                correct += 1

            if result["predicted_index"] == 1:
                pred_value_1 += 1

    elif dataset == "vqa":
        for result in results:
            if result["predicted_label"] == 1:
                correct += 1
                pred_value_1 += 1

    if correct == 0:
        return 0.0, pred_value_1, len(results)
    else:
        accuracy = correct / len(results)
        return accuracy, pred_value_1, len(results)
