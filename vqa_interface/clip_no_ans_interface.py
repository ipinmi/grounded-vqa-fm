import torch
import clip
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocessor = clip.load("ViT-B/32", device=device)


# NOTE: Adjust code (defined answer types)
def test_CLIP_on_VCR(dataloader: DataLoader):
    """
    Test the CLIP model on a batch of VCR data without passing the answer to the model.

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

            results[annot_id[0]] = {
                "question": questions[0],  # Assuming one question per image
                "predicted_answer": answers[highest_prob_idx],  # The predicted answer
                "predicted_index": int(predicted_labels + 1),  # Index to label
                "correct_answer": answers[labels.index(1)],  # The correct answer
                "correct_index": int(labels.index(1) + 1),  # Index to label
            }

    return results


def test_CLIP_on_VQA(dataloader: DataLoader, dataset: Dataset, save_tensor=True):
    """
    Test the CLIP model on a batch of VQA V2 data without passing the answer to the model.

    Args:
    dataloader (DataLoader): A DataLoader object containing the VQA data.
    dataset (Dataset): A VQADataset object containing the VQA data and answer types.
    save_tensor (bool): A flag to create and save the tensor representations of answers, if not already saved.

    Returns:
    results (list): A list of dictionaries containing the following keys:
        - annot_id (int): The annotation ID.
        - question (str): The question.
        - predicted_label (list): The predicted label = 1 if the probability is greater than 0.5. Otherwise, 0.
    """

    results = []
    total_correct = 0
    type_correct = {key: 0 for key in dataset.answers_by_type.keys()}

    # answer types: yes/no, number, other
    possible_answers = dataset.answers_by_type

    # Tokenize and normalize the answer types
    if save_tensor:
        with torch.no_grad():
            yes_no_ans = clip.tokenize(possible_answers["yes/no"]).to(device)
            number_ans = clip.tokenize(possible_answers["number"]).to(device)
            other_ans = clip.tokenize(possible_answers["other"]).to(device)

            yes_no_features = model.encode_text(yes_no_ans)
            number_features = model.encode_text(number_ans)
            other_features = model.encode_text(other_ans)

            yes_no_features /= yes_no_features.norm(dim=-1, keepdim=True)
            number_features /= number_features.norm(dim=-1, keepdim=True)
            other_features /= other_features.norm(dim=-1, keepdim=True)

            # Save the tensor representations
            torch.save(yes_no_features, "features/yes_no_features.pt")
            torch.save(number_features, "features/number_features.pt")
            torch.save(other_features, "features/other_features.pt")

    else:
        yes_no_features = torch.load("features/yes_no_features.pt").to(device)
        number_features = torch.load("features/number_features.pt").to(device)
        other_features = torch.load("features/other_features.pt").to(device)

    for idx, batch in enumerate(tqdm(dataloader)):
        annot_ids = batch["annot_id"].detach().numpy()
        questions = batch["question"]
        answers = batch["answer"]
        image_paths = batch["image_path"]
        answer_types = batch["answer_type"]
        answer_index = batch["answer_idx"]

        # Prepare the text inputs (questions) and image inputs
        question_toks = clip.tokenize(questions).to(device)
        images = [Image.open(image_path) for image_path in image_paths]
        image_features = torch.stack([preprocessor(i) for i in images]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_features)
            question_features = model.encode_text(question_toks)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        question_features /= question_features.norm(dim=-1, keepdim=True)

        # Combine and normalize the joint features
        joint_features = question_features + image_features
        joint_features /= joint_features.norm(dim=-1, keepdim=True)

        # Iterate through each question in the batch
        batch_results = []
        for i, answer_type in enumerate(answer_types):
            if answer_type == "yes/no":
                similarity = joint_features[i] @ yes_no_features.T
            elif answer_type == "number":
                similarity = joint_features[i] @ number_features.T
            elif answer_type == "other":
                similarity = joint_features[i] @ other_features.T
            else:
                raise ValueError(f"Unknown answer type: {answer_type}")

            # Extract the maximum similarity and the predicted index
            max_similarity, pred_idx = similarity.max(dim=-1)

            if pred_idx.item() == answer_index[i]:
                type_correct[answer_type] += 1
                total_correct += 1

            batch_results.append(
                {
                    "annot_id": annot_ids[i],
                    "question": questions[i],
                    "predicted_idx": pred_idx.item(),
                    "similarity": max_similarity.item(),
                    "answer_type": answer_type,
                }
            )

        results.extend(batch_results)

        # calculate accuracy over each type of answer and overall
        accuracy = total_correct / len(results)

        for typ in type_correct:
            type_accuracy = type_correct[typ] / len(possible_answers[typ])
            print(f"Accuracy for {typ} answers: {type_accuracy}")
            print(f"Total correct for {typ} answers: {type_correct[typ]}")
            print("--------------------------------------------")

        print(f"Total accuracy: {accuracy}")
    return results
