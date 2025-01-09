import torch
import clip
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocessor = clip.load("ViT-B/32", device=device)


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
    correct = 0

    for idx, batch in enumerate(tqdm(dataloader)):

        annot_id, image_paths, questions, answers, labels, detections = batch

        # Assuming one image per batch, open and preprocess it
        image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

        with torch.no_grad():
            # Prepare and encode the possible multichoice answers (expected: 4)
            encoded_answers = clip.tokenize(answers).to(device)  # shape: (4, 77)
            answers_features = model.encode_text(encoded_answers).to(
                device
            )  # shape: (4, 512)
            answers_features /= answers_features.norm(dim=-1, keepdim=True)

            # encode the question
            encoded_questions = clip.tokenize(questions[0]).to(device)  # shape: (1, 77)

            # Create inputs for the model
            image_features = model.encode_image(image)
            question_features = model.encode_text(encoded_questions)

            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            question_features /= question_features.norm(dim=-1, keepdim=True)

            # Combine and normalize the joint features
            joint_features = question_features + image_features
            joint_features /= joint_features.norm(dim=-1, keepdim=True)

            # Compute the similarity between the joint and answers features
            similarity_array = joint_features @ answers_features.T

            # Extract the maximum similarity and the predicted index
            max_similarity, pred_idx = similarity_array.max(dim=-1)

            if pred_idx.item() == labels.index(1):
                correct += 1

            results[annot_id[0]] = {
                "question": questions[0],  # Assuming one question per image
                "predicted_answer": answers[pred_idx],  # The predicted answer
                "predicted_index": int(pred_idx.item() + 1),  # Index to label
                "similarity": round(max_similarity.item(), 4),  # predicted similarity
                "correct_answer": answers[labels.index(1)],  # The correct answer
                "correct_index": int(labels.index(1) + 1),  # Index to label
            }

    # Calculate accuracy
    total_accuracy = correct / len(results)

    print(f"Overall accuracy: {total_accuracy:.2%} ({correct}/{len(results)})")

    return results, total_accuracy


def test_CLIP_on_VQA(dataloader: DataLoader, dataset: Dataset):
    """
    Test the CLIP model on a batch of VQA V2 data without passing the answer to the model.
    The question-answer pairs are grouped by answer type and the model is tested on each group.
    The image + questions are encoded and compared for the most similar answer type.

    Args:
    dataloader (DataLoader): A DataLoader object containing the VQA data.
    dataset (Dataset): A VQADataset object containing the VQA data and answer types.

    Returns:
    results (list): A list of dictionaries containing the following keys:
        - annot_id (int): The annotation ID.
        - question (str): The question.
        - predicted_idx (list): The predicted index.
        - similarity (float): The similarity score.
        - answer_type (str): The answer type.
    """

    results = []
    total_correct = 0
    type_correct = {key: 0 for key in dataset.answers_by_type.keys()}
    type_total = {key: 0 for key in dataset.answers_by_type.keys()}

    # Answer types: yes/no, number, other
    possible_answers = dataset.answers_by_type

    # Tokenize and normalize the answer types
    with torch.no_grad():
        yes_no_ans = clip.tokenize(possible_answers["yes/no"]).to(
            device
        )  # shape: [no_of_possible_ans, 77]
        number_ans = clip.tokenize(possible_answers["number"]).to(device)
        other_ans = clip.tokenize(possible_answers["other"]).to(device)

        yes_no_features = model.encode_text(yes_no_ans)  # shape: [batch_size, 512]
        number_features = model.encode_text(number_ans)
        other_features = model.encode_text(other_ans)

        yes_no_features /= yes_no_features.norm(dim=-1, keepdim=True)
        number_features /= number_features.norm(dim=-1, keepdim=True)
        other_features /= other_features.norm(dim=-1, keepdim=True)

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
        for i, answer_type in enumerate(answer_types):
            type_total[answer_type] += 1  # Count occurrences of each type

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

            # Check if the prediction matches the ground truth
            if pred_idx.item() == answer_index[i]:
                type_correct[answer_type] += 1
                total_correct += 1

            results.append(
                {
                    "annot_id": annot_ids[i],
                    "question": questions[i],
                    "predicted_idx": pred_idx.item(),
                    "similarity": max_similarity.item(),
                    "answer_type": answer_type,
                }
            )

    # Calculate accuracy for each type and overall
    total_accuracy = total_correct / len(results)

    for typ in type_correct:
        if type_total[typ] > 0:  # Avoid division by zero
            type_accuracy = type_correct[typ] / type_total[typ]
            print(
                f"Accuracy for {typ} answers: {type_accuracy:.2%} ({type_correct[typ]}/{type_total[typ]})"
            )
        else:
            print(f"No questions of type {typ} were found.")

    print(f"Overall accuracy: {total_accuracy:.2%} ({total_correct}/{len(results)})")

    return results, total_accuracy
