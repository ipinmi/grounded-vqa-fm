import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader


def run_CLIP_batch(dataloader: DataLoader):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    results = {}

    for idx, batch in enumerate(dataloader):
        image_paths, questions, answers, labels = batch

        # Assuming one image per batch, open it
        image = Image.open(image_paths[0])

        # Prepare the text inputs (questions paired with answers)
        question_answers = [
            f"question: {question} answer: {answer}"
            for question, answer in zip(questions, answers)
        ]

        with torch.no_grad():
            # Create inputs for the model
            inputs = processor(
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

            # Map back to image ID
            image_id = image_paths[0].replace("data/vcr1images/", "")
            highest_prob_idx = np.argmax(predicted_probs)

            # Store results
            results[image_id] = {
                "question": questions[0],  # Assuming one question per image
                "predicted_answer": answers[highest_prob_idx],  # The predicted answer
                "predicted_index": predicted_labels + 1,  # Index to label
                "predicted_label": predicted_probs.tolist(),  # Probabilities for each answer
                "expected_label": labels,
                "correct_answer": answers[labels.index(1)],  # The correct answer
            }

    return results
