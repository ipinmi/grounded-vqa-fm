import argparse
import matplotlib.pyplot as plt
import json
import subprocess


##MODULES
from vqa_interface.clip_interface import run_CLIP_batch, eval_on_accuracy
from vcr_data.vcr_dataloader import (
    VCRDataExtractor,
    VCRDataset,
    VCRDataLoader,
    BatchSampler,
)

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


args = parser.parse_args()
VCR_ANNOTS_DIR = args.annots_dir
VCR_IMAGES_DIR = args.image_dir
results_path = args.results_path

subprocess.run(["mkdir", "-p", results_path])


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    extracted_vcr = VCRDataExtractor(
        VCR_ANNOTS_DIR,
        VCR_IMAGES_DIR,
        mode="answer",
        split="train",
        only_use_relevant_dets=False,
    )
    dataset = VCRDataset(extracted_vcr, "vqa")
    batch_sampler = BatchSampler(dataset, batch_size=4)
    dataloader = VCRDataLoader(dataset, batch_sampler=batch_sampler)

    # Run the CLIP model
    vqa_results = run_CLIP_batch(dataloader)
    save_json(vqa_results, "results/clip_vqa_results.json")
    # print(vqa_results)

    # Evaluate the model
    accuracy, pred_value_1, total = eval_on_accuracy(vqa_results)
    print(f"Accuracy: {accuracy}")
    print(f"No of times Predicted value is 1: {pred_value_1}")
    print(f"Total: {total}")


if __name__ == "__main__":
    main()


# python3 -W ignore main.py --annots_dir data/vcr1annots --image_dir data/vcr1images
