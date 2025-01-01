import argparse
import matplotlib.pyplot as plt
import json
import subprocess
from torch.utils.data import DataLoader


##MODULES
from vqa_interface.clip_orig_interface import (
    run_CLIP_on_VCR,
    run_CLIP_on_VQA,
    eval_on_accuracy,
)

from vqa_interface.clip_no_ans_interface import test_CLIP_on_VQA

from vcr_data.vcr_dataloader import (
    VCRDataExtractor,
    VCRDataset,
    VCRDataLoader,
    BatchSampler,
)

from vqa_data.vqa_dataloader import load_vqa_data, VQADataset

# from clip_detector.object_detector import *

####
# Arguments for data preprocessing and loading
parser = argparse.ArgumentParser()
parser.add_argument(
    "--annots_dir",
    help="Directory path for annotations where train.jsonl, val.jsonl, test.jsonl are stored",
    default="data/vcr1annots",
    required=True,
)
parser.add_argument(
    "--image_dir",
    help="Directory path for images holding their segmentations, boxes, and image files",
    default="data/vcr1images",
    required=True,
)
parser.add_argument(
    "--results_path",
    help="Directory path for saving the model results",
    default="results",
)

parser.add_argument(
    "--dataset",
    help="Dataset to be used",
    default="vcr",
    required=True,
)

parser.add_argument(
    "--ans_mode",
    help="Answer or No Answer mode",
    default="answer",
    required=True,
)

args = parser.parse_args()
ANNOTS_DIR = args.annots_dir
IMAGES_DIR = args.image_dir
results_path = args.results_path
dataset_type = args.dataset
answer_mode = args.ans_mode

# make results directory
subprocess.run(["mkdir", "-p", results_path])

# make feature directory
# subprocess.run(["mkdir", "-p", "vqa_interface/features"])


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, default=int)


def vcr_main():
    # Extract data from VCR and create a dataloader
    extracted_vcr = VCRDataExtractor(
        ANNOTS_DIR,
        IMAGES_DIR,
        mode="answer",
        split="val",
        only_use_relevant_dets=False,
    )
    dataset = VCRDataset(extracted_vcr, "vqa")
    batch_sampler = BatchSampler(dataset, batch_size=4)
    dataloader = VCRDataLoader(dataset, batch_sampler=batch_sampler)

    # Run the CLIP model
    print("Running CLIP on VCR data...\n")
    vcr_results = run_CLIP_on_VCR(dataloader)
    save_json(vcr_results, "results/clip_vcr_results.json")
    # print(vqa_results)

    # Evaluate the model
    accuracy, pred_value_1, total = eval_on_accuracy(vcr_results, dataset_type)
    print(f"Accuracy: {accuracy}")
    print(f"No of times Predicted value is 1: {pred_value_1}")
    print(f"Total: {total}")


def vqa_main():
    batchSize = 4
    qa_pairs, possible_answers_by_type = load_vqa_data(ANNOTS_DIR)
    # Create dataset and dataloader
    dataset = VQADataset(qa_pairs, ANNOTS_DIR, possible_answers_by_type)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    # Run the CLIP model on VQA V2 data
    print("Running CLIP on VQA V2 data...\n")
    if answer_mode == "answer":
        vqa_results = run_CLIP_on_VQA(dataloader)
        save_json(vqa_results, "results/clip_vqa_results.json")

        # Evaluate the model
        accuracy = eval_on_accuracy(vqa_results, dataset_type)
        accuracy, pred_value_1, total = eval_on_accuracy(vqa_results, dataset_type)
        print(f"Accuracy: {accuracy}")
        print(f"Number of accurate results: {pred_value_1}")
        print(f"Total: {total}")
    elif answer_mode == "no_ans":
        vqa_results = test_CLIP_on_VQA(dataloader, dataset=dataset, save_tensor=True)
        save_json(vqa_results, "results/clip_vqa_results_no_ans.json")


if __name__ == "__main__":
    if dataset_type == "vcr":
        vcr_main()

    elif dataset_type == "vqa":
        vqa_main()


# python3 -W ignore main.py --annots_dir data/vcr1annots --image_dir data/vcr1images --dataset vcr --ans_mode no_ans
# python3 -W ignore main.py --annots_dir data/vqa_v2 --image_dir data/vqa_v2 --dataset vqa --ans_mode no_ans
