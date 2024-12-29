import json
from torch.utils.data import Dataset, DataLoader


# Load questions and annotations (answers)
def load_vqa_data(filepath: str):
    """
    Load the VQA dataset from the given filepath for the downloaded images.

    Args:
        filepath: path to the directory containing the VQA dataset

    Returns:
        qa_pairs: a dictionary of dictionaries, where each dictionary contains:
            - 'image_id': int
            - 'question': str
            - 'answer': str (primary answer to the question)
    """
    with open(f"{filepath}/v2_OpenEnded_mscoco_val2014_questions.json", "r") as f:
        questions_data = json.load(f)

    with open(f"{filepath}/v2_mscoco_val2014_annotations.json", "r") as z:
        annotations_data = json.load(z)

    # Combine questions and annotations into a dictionary
    qa_pairs = {}
    for q, a in zip(questions_data["questions"], annotations_data["annotations"]):
        question_id = q["question_id"]
        qa_pairs[question_id] = {
            "image_id": q["image_id"],
            "question": q["question"],
            "answer": a["multiple_choice_answer"],  # Use the primary answer
        }

    return qa_pairs


class VQADataset(Dataset):
    def __init__(self, data, filepath):
        """
        Args:
            data: a dictionary of dictionaries, where each dictionary contains:
                - 'question': str
                - 'answer': str (answer to the question)
                - 'image_id':id of the image
            filepath: path to the directory containing the images
        """
        self.data = []

        for key, instance in data.items():
            annot_id = key
            image_id = str(instance["image_id"])
            question = instance["question"]
            answer = instance["answer"]

            # Add the image full path
            if len(str(image_id)) < 12:
                image_id = "0" * (12 - len(image_id)) + image_id

                assert len(image_id) == 12

            image_path = f"{filepath}/val2014/COCO_val2014_{image_id}.jpg"

            self.data.append(
                {
                    "annot_id": annot_id,
                    "question": question,
                    "answer": answer,
                    "image_path": image_path,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def custom_collate_fn(batch):
    # Since batch is a list of dictionaries, return the batch as is
    return batch


def main():
    file_path = "data/vqa_v2"
    batchSize = 1
    qa_pairs = load_vqa_data(file_path)
    # Create dataset and dataloader
    val_data = VQADataset(qa_pairs, file_path)
    val_dataloader = DataLoader(val_data, batch_size=batchSize, shuffle=True)

    return val_dataloader
