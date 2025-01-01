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
    possible_answers_by_type = {}
    i = 0
    for q, a in zip(questions_data["questions"], annotations_data["annotations"]):
        question_id = q["question_id"]
        qa_pairs[question_id] = {
            "image_id": q["image_id"],
            "question": q["question"],
            "question_type": a["question_type"],
            "answer": a["multiple_choice_answer"],  # Use the primary answer
            "answer_type": a["answer_type"],
        }

        if a["answer_type"] not in possible_answers_by_type:
            possible_answers_by_type[a["answer_type"]] = []

        possible_answers_by_type[a["answer_type"]].append(a["multiple_choice_answer"])
    # set of all unique answers
    for key in possible_answers_by_type:
        possible_answers_by_type[key] = list(set(possible_answers_by_type[key]))

    return qa_pairs, possible_answers_by_type


class VQADataset(Dataset):
    def __init__(self, data, filepath, answers_by_type):
        """
        Args:
            data: a dictionary of dictionaries, where each dictionary contains:
                - 'question': str
                - 'answer': str (answer to the question)
                - 'image_id':id of the image
            filepath: path to the directory containing the images
            answers_by_type: a dictionary containing the set of possible answers for each answer type
        """
        self.data = data
        self.filepath = filepath
        self.dataset = []

        # Create answer vocabulary based on the answer type
        self.answers_by_type = answers_by_type
        self.index_answer, self.answer_index = {}, {}

        for answer_type, answers in self.answers_by_type.items():
            self.index_answer[answer_type] = dict(enumerate(answers))
            self.answer_index[answer_type] = {
                answer: idx for idx, answer in enumerate(answers)
            }

        # Create the dataset
        for key, instance in self.data.items():
            annot_id = key
            image_id = str(instance["image_id"])
            question = instance["question"]
            answer = instance["answer"]
            answer_type = instance["answer_type"]
            answer_idx = self.answer_index[answer_type][answer]

            # Add the image full path
            if len(str(image_id)) < 12:
                image_id = "0" * (12 - len(image_id)) + image_id

                assert len(image_id) == 12

            image_path = f"{self.filepath}/val2014/COCO_val2014_{image_id}.jpg"

            self.dataset.append(
                {
                    "annot_id": annot_id,
                    "question": question,
                    "answer": answer,
                    "image_path": image_path,
                    "answer_type": answer_type,
                    "answer_idx": answer_idx,
                }
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def custom_collate_fn(batch):
    # Since batch is a list of dictionaries, return the batch as is
    return batch
