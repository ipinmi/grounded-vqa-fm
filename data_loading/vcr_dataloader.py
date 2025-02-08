# Contains refactored code from Dataset (VCRDataExtractor and index_to_names). Attribution: r2c (VCR dataset)

# USAGE: python3 dataloader/data.py --annots_dir data/vcr1annots --image_dir data/vcr1images
import os
import json
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random

# Gender-neutral names for replacing the label for "person"
GENDER_NEUTRAL_NAMES = [
    "Casey",
    "Riley",
    "Jessie",
    "Jackie",
    "Avery",
    "Jaime",
    "Peyton",
    "Kerry",
    "Jody",
    "Kendall",
    "Peyton",
    "Skyler",
    "Frankie",
    "Pat",
    "Quinn",
]


def index_to_names(sentence, object_type_list, rel_detection_indices, padding_idx=-1):
    """
    Converts the object detection labels to gender-neutral names
    for processing by the vision-language model, adding "and" between two person objects.

    Args:
    sentence: A string containing the object detection labels as an embedded list
    object_type_list: A list of object types detected in the image
    rel_detection_indices: The indices of the relevant or all detected objects (
                        Mapping of the old ID -> new ID based on the number of detections to use)

    Returns:
    Tuple of two lists:
        - Converted sentence as a list of names.
        - Corresponding tags as a list of new indices.
    """
    new_sentence = []
    new_tags = []

    last_was_person = False

    for obj_detect in sentence:
        if isinstance(obj_detect, list):
            for obj_index in obj_detect:
                obj_type = object_type_list[obj_index]
                new_idx = rel_detection_indices[obj_index]

                if new_idx < 0:
                    raise ValueError(
                        f"Invalid index for object detected. {sentence} {rel_detection_indices}"
                    )

                # Replace the label for "person" with the gender-neutral name
                if obj_type == "person":
                    name = GENDER_NEUTRAL_NAMES[new_idx % len(GENDER_NEUTRAL_NAMES)]
                    if last_was_person:
                        new_sentence.append(
                            "and"
                        )  # Add "and" between consecutive persons
                    new_sentence.append(name)
                    last_was_person = True
                else:
                    new_sentence.append(obj_type)
                    last_was_person = False

                new_tags.append(new_idx)
        else:
            new_sentence.append(obj_detect)
            new_tags.append(padding_idx)
            last_was_person = False

    return new_sentence, new_tags


class VCRDataExtractor(Dataset):
    def __init__(
        self,
        annots_dir,
        image_dir,
        mode="answer",
        split="val",
        only_use_relevant_dets=False,
    ):
        """
        Args:
        split: train, val, val_test (split from val) or test
        mode: answer or rationale (Q-A or QA-R tasks)
        only_use_relevant_dets: If True, only use the relevant detections
        """
        self.mode = mode
        self.split = split
        self.use_relevant_dets = only_use_relevant_dets
        self.annots_dir = annots_dir
        self.image_dir = image_dir

        # Load the annotations for the split
        try:
            with open(
                os.path.join(annots_dir, "{}.jsonl".format(split)), "r"
            ) as jsonl_file:
                self.data = [json.loads(line) for line in jsonl_file]
        except FileNotFoundError:
            print(
                f"Warning: Annotation file '{split}.jsonl' not found in '{annots_dir}'. Skipping data loading."
            )
            self.data = []

        # Load COCO ontology for object detection labels
        try:
            with open(
                os.path.join(self.annots_dir, "cocoontology.json"),
                "r",
            ) as f:
                coco = json.load(f)
            self.coco_objects = ["__background__"] + [
                x["name"] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))
            ]
            self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}
        except FileNotFoundError:
            print(
                "Warning: 'cocoontology.json' not found. Object detection labels will be empty."
            )
            self.coco_objects = []
            self.coco_obj_to_ind = {}

    def __len__(self):
        return len(self.data)

    def _get_dets_to_use(self, item):
        """
        Get the indices of the relevant or all detected objects
        based on the number of detections to use.

        Args:
        item: Sentence with object detection labels

        Returns:
        Tuple of two lists:
            - Indices of the relevant detections
            - Mapping of the old ID -> new ID based on the number of detections to use
        """
        # Get the question and answer choices
        question = item["question"]
        answer_choices = item["answer_choices"]

        if self.use_relevant_dets:
            # an array of booleans based on the number of objects detected
            dets2use = np.zeros(len(item["objects"]), dtype=bool)

            # an array of booleans based on the number of people detected
            people = np.array([x == "person" for x in item["objects"]], dtype=bool)

            # Concatenate the question and answer choices
            for sent in answer_choices + [question]:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item["objects"]):
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ("everyone", "everyones"):
                        dets2use |= people  # everyone is a person
            if not dets2use.any():
                dets2use |= people  # if no detections, use all people
        else:
            dets2use = np.ones(len(item["objects"]), dtype=bool)  # use all detections

        # the relevant detections to be used
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item["objects"]), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        """
        # If we add the image as an extra box then the 0th will be the image.
        if add_image_as_a_box:
                old_det_to_new_ind[dets2use] += 1"""
        old_det_to_new_ind = old_det_to_new_ind.tolist()

        return dets2use, old_det_to_new_ind

    def __getitem__(self, idx):
        """
        Creates the instance dictionary based on the index.

        Args:
        idx: Index of the instance

        Returns:
        Dictionary containing the instance information or None if the instance cannot be created.
        """
        item = deepcopy(self.data[idx])

        try:
            dets2use, old_det_to_new_ind = self._get_dets_to_use(item)

            # for the QA-R task, append the correct answer choice to the question
            if self.mode == "rationale":
                conditioned_label = item["answer_label"]
                item["question"] += item["answer_choices"][conditioned_label]
                answer_choices = item["rationale_choices"]

            # for the Q-A task, use the answer choices
            elif self.mode == "answer":
                answer_choices = item["answer_choices"]

            # Convert the question and answer choices
            converted_questions, converted_question_tags = index_to_names(
                item["question"],
                item["objects"],
                old_det_to_new_ind,
                padding_idx=-1,
            )

            converted_answers, converted_answer_tags = zip(
                *[
                    index_to_names(
                        answer,
                        item["objects"],
                        old_det_to_new_ind,
                        padding_idx=-1,
                    )
                    for answer in answer_choices
                ]
            )

            instance_dict = {}  # Dictionary to store the instance

            ########## IMAGE INFORMATION ##########

            # Get image and its metadata paths
            image_path = os.path.join(self.image_dir, item["img_fn"])
            instance_dict["image_path"] = image_path

            img_metadata_path = os.path.join(self.image_dir, item["metadata_fn"])

            # Load metadata for the image
            try:
                with open(img_metadata_path, "r") as img_metadata_file:
                    img_metadata = json.load(img_metadata_file)
            except FileNotFoundError:
                print(
                    f"Warning: Metadata file '{img_metadata_path}' not found. Skipping instance at index {idx}."
                )
                return None

            # Remove final dimension (detection confidence) from boxes
            boxes = np.array(img_metadata["boxes"])[dets2use, :-1]

            # Object categories based on COCO ontology
            instance_dict["objects_cats"] = [item["objects"][i] for i in dets2use]

            obj_labels = [
                self.coco_obj_to_ind[item["objects"][i]] for i in dets2use.tolist()
            ]

            # NOTE: adding the whole image as an object
            objects_tensors = [torch.Tensor(obj_labels) for x in obj_labels]
            instance_dict["objects"] = torch.stack(objects_tensors, dim=0)

            # NOTE: padding the bounding boxes (-1) in dataloader
            instance_dict["boxes"] = torch.Tensor(boxes)

            ########## QUESTION-ANSWER INFORMATION ##########

            # Add the question, question tags, answers, and answer tags to the instance
            instance_dict["question"] = converted_questions
            instance_dict["question_tags"] = converted_question_tags
            instance_dict["answers"] = converted_answers
            instance_dict["answer_tags"] = converted_answer_tags

            # Add the label for the instance based on answers or rationale mode
            if self.split != "val_test":
                instance_dict["label"] = item["{}_label".format(self.mode)]

            instance_dict["metadata"] = {
                "index": idx,
                "annot_id": item["annot_id"],
                "movie": item["movie"],
                "img_fn": item["img_fn"],
                "question_number": item["question_number"],
            }

            return instance_dict

        except Exception as e:
            print(f"Error processing instance at index {idx}: {e}")
            return None


class VCRDataset(Dataset):
    def __init__(self, extractor, objective="vqa", load_all: bool = True, size=None):
        """
        Args:
            extractor: An instance of VCRDataExtractor where each instance is a dictionary contains:
                - 'question': str
                - 'answers': list of 4 answers
                - 'label': index of the correct answer (0 to 3)
            objective: zero shot objective
                - 'vqa' (for visual question answering)
                - 'objdet' (for object detection)
            load_all: If True, load the entire dataset
            size: Number of instances to load
        """
        self.extractor = extractor
        self.data = []
        self.fields_to_add = ["objects", "objects_cats"]
        self.size = size if size is not None else len(self.extractor)

        for instance in self.extractor:
            if instance is None:
                continue
            annot_id = instance["metadata"]["annot_id"]
            image_path = instance["image_path"]
            question = instance["question"]
            answers = instance["answers"]
            label = instance["label"]
            bounding_boxes = instance["boxes"]

            ans_length = any(
                len(answer) > 77 for answer in answers
            )  # limited for qa task
            question_length = any(
                len(answer) > 77 for answer in answers
            )  # limited for qa-r task

            if ans_length or question_length:
                continue

            self.data.extend(
                {
                    "annot_id": annot_id,
                    "question": question,
                    "answer": answer,
                    "label": int(idx == label),
                    "image_path": image_path,
                    "label_index": label,
                    "boxes": bounding_boxes,
                    **(
                        {
                            field: instance[field]
                            for field in self.fields_to_add
                            if field in instance
                        }
                        if objective == "objdet"
                        else {}
                    ),
                }
                for idx, answer in enumerate(answers)
            )

        if not load_all:
            # Limit the dataset size
            if self.size > len(self.data):
                raise ValueError(
                    f"Dataset size {self.size} is greater than the available data {len(self.data)}."
                )
            self.data = self.data[: self.size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size

        # Ensure no remainder
        assert (
            len(dataset) % batch_size == 0
        ), "Dataset size must be divisible by batch size."

    def __iter__(self):
        # Generate batch indices
        batch_indices = list(range(self.num_batches))
        random.shuffle(batch_indices)  # Shuffle batch order

        for batch_idx in batch_indices:
            # Generate indices for the current batch
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            yield list(
                range(start_idx, end_idx)
            )  # Yield indices in order within the batch

    def __len__(self):
        return self.num_batches


class VCRDataLoader(DataLoader):
    def __init__(self, dataset, batch_sampler):
        """
        Args:
            dataset: An instance of the VCRDataset class
            batch_sampler: An instance of BatchSampler for custom batch shuffling
        """
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,  # Pass BatchSampler here
            shuffle=False,  # Disable shuffle as batch_sampler handles it
            collate_fn=self._collate_fn,  # Use the custom collate function
        )

    @staticmethod
    def _collate_fn(batch):
        """
        Custom collate function to handle batching.
        Args:
            batch: List of individual items
        Returns:
            Tuple (image_paths, questions, answers, labels, optional_fields)
        """

        def collect_unique(field):
            return list(
                {
                    tuple(item[field]) if isinstance(item[field], list) else item[field]
                    for item in batch
                    if field in item
                }
            )

        annot_id = collect_unique("annot_id")
        questions = [" ".join(item["question"]) for item in batch]
        answers = [" ".join(item["answer"]) for item in batch]
        labels = [item["label"] for item in batch]
        image_paths = collect_unique("image_path")
        optional_fields = {
            "boxes": collect_unique("boxes"),
            "objects": collect_unique("objects"),
            "objects_cats": collect_unique("objects_cats"),
        }

        return annot_id, image_paths, questions, answers, labels, optional_fields
