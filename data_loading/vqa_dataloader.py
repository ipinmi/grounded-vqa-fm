import json
from torch.utils.data import Dataset, DataLoader
import random

# Load questions and annotations (answers)
from collections import defaultdict, Counter
import json


def load_vqa_data(
    filepath: str,
    split: str,
    top_k: int = 1000,
    max_pairs: int = 3999,
    load_all: bool = False,
):
    """
    Load the VQA dataset from the given filepath for the downloaded images.

    Args:
        filepath: Path to the directory containing the VQA dataset.
        split: The split of the dataset to load (e.g., 'val', 'train', 'test').
        top_k: The number of most frequent answers to keep for each type.
        max_pairs: The maximum number of question-answer pairs to include.
        load_all: If True, load all question-answer pairs without filtering.

    Returns:
        qa_pairs: A dictionary where each key is a question ID, and the value is a dictionary containing:
            - 'image_id': int
            - 'question': str
            - 'answer': str (primary answer to the question)
            - 'question_type': str
            - 'answer_type': str
        possible_answers_by_type: A dictionary mapping answer types to their top-k answers.
    """
    # Load questions and annotations
    with open(f"{filepath}/v2_OpenEnded_mscoco_{split}2014_questions.json", "r") as f:
        questions_data = json.load(f)

    with open(f"{filepath}/v2_mscoco_{split}2014_annotations.json", "r") as z:
        annotations_data = json.load(z)

    # Initialize data structures
    qa_pairs = {}
    answers_by_type = defaultdict(list)
    all_answers = []

    # Combine questions and annotations
    for q, a in zip(questions_data["questions"], annotations_data["annotations"]):
        question_id = q["question_id"]
        qa_pairs[question_id] = {
            "image_id": q["image_id"],
            "question": q["question"],
            "question_type": a["question_type"],
            "answer": a["multiple_choice_answer"],
            "answer_type": a["answer_type"],
        }
        answers_by_type[a["answer_type"]].append(a["multiple_choice_answer"])

    if load_all:
        all_answers = list(
            set(
                answers_by_type["yes/no"]
                + answers_by_type["number"]
                + answers_by_type["other"]
            )
        )
        return qa_pairs, answers_by_type, all_answers

    else:
        # Select the top-k answers for each type
        possible_answers_by_type = {
            answer_type: [answer for answer, _ in Counter(answers).most_common(top_k)]
            for answer_type, answers in answers_by_type.items()
        }

        # Filter qa_pairs to include only those with top-k answers
        filtered_qa_pairs = {
            q_id: data
            for q_id, data in qa_pairs.items()
            if data["answer"] in possible_answers_by_type[data["answer_type"]]
        }

        # Select based on max_pairs
        reduced_qa_pairs = dict(list(filtered_qa_pairs.items())[:max_pairs])

        # Get the joint set of possible answers
        for answers in possible_answers_by_type.values():
            all_answers.extend(answers)

        random.shuffle(all_answers)

        return reduced_qa_pairs, possible_answers_by_type, all_answers


class VQADataset(Dataset):
    def __init__(self, data, split, filepath, answers_by_type, all_answers):
        """
        Args:
            data: a dictionary of dictionaries, where each dictionary contains:
                - 'question': str
                - 'answer': str (answer to the question)
                - 'image_id':id of the image
            split: The split of the dataset to load (e.g., 'val', 'train', 'test').
            filepath: path to the directory containing the images
            answers_by_type: a dictionary containing the set of possible answers for each answer type
            all_answers: a list of all possible answers from all answer types
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
            answer_type_idx = self.answer_index[answer_type][answer]
            answer_idx = all_answers.index(answer)

            # Add the image full path
            if len(str(image_id)) < 12:
                image_id = "0" * (12 - len(image_id)) + image_id

                assert len(image_id) == 12

            image_path = f"{self.filepath}/{split}2014/COCO_{split}2014_{image_id}.jpg"

            self.dataset.append(
                {
                    "annot_id": annot_id,
                    "question": question,
                    "answer": answer,
                    "image_path": image_path,
                    "answer_type": answer_type,
                    "answer_type_idx": answer_type_idx,
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
