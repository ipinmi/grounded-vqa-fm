import argparse
import json
import subprocess
from torch.utils.data import DataLoader


##MODULES
from data_loading.vcr_dataloader import (
    VCRDataExtractor,
    VCRDataset,
    VCRDataLoader,
    BatchSampler,
)

from data_loading.vqa_dataloader import load_vqa_data, VQADataset

# Baseline VQA CLIP models on VCR and VQA V2
from zero_shot_clip.clip_ans_interface import (
    run_CLIP_on_VCR,
    run_CLIP_on_VQA,
)

from zero_shot_clip.clip_no_ans_interface import test_CLIP_on_VQA, test_CLIP_on_VCR

####
# Arguments for data preprocessing and loading
parser = argparse.ArgumentParser()
parser.add_argument(
    "--annots_dir",
    help="Directory path for annotations where the questions, answers and image ids are stored",
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


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, default=int)


def vcr_main():
    # Extract data from VCR and create a dataloader
    num_qa_pairs = 1000
    extracted_vcr = VCRDataExtractor(
        ANNOTS_DIR,
        IMAGES_DIR,
        mode="answer",
        split="val",
        only_use_relevant_dets=True,
    )
    dataset = VCRDataset(extracted_vcr, "vqa", size=num_qa_pairs)
    batch_sampler = BatchSampler(dataset, batch_size=4)
    dataloader = VCRDataLoader(dataset, batch_sampler=batch_sampler)

    # Run the CLIP model
    print("Running CLIP on VCR data...\n")
    if answer_mode == "answer":
        vcr_results, accuracy = run_CLIP_on_VCR(dataloader)
        save_json(vcr_results, f"{results_path}/clip_vcr_results.json")
        # print(vcr_results)
    elif answer_mode == "no_ans":
        vcr_results, accuracy = test_CLIP_on_VCR(dataloader)
        save_json(vcr_results, f"{results_path}/clip_vcr_results_no_ans.json")
        # print(vcr_results)

    # Save the evaluation results
    with open(f"{results_path}/evaluation_results_{dataset_type}.txt", "a") as f:
        f.write(f"Mode: CLIP Zero shot Model Evaluation in {answer_mode} \n")
        f.write(f"Dataset: VCR \n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Validation number of batches: {len(vcr_results)} \n")
        f.write(f"Number of QA pairs: {num_qa_pairs} \n")
        f.write("\n")


def vqa_main():
    batchSize = 64
    top_k_ans = 1000
    num_qa_pairs = 100000

    # Testing over the validation set of VQA V2
    val_qa_pairs, val_possible_answers_by_type, val_answers = load_vqa_data(
        ANNOTS_DIR, split="val", top_k=top_k_ans, max_pairs=num_qa_pairs, load_all=True
    )
    val_dataset = VQADataset(
        val_qa_pairs,
        split="val",
        filepath=ANNOTS_DIR,
        answers_by_type=val_possible_answers_by_type,
        all_answers=val_answers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)

    # Run the CLIP model on VQA V2 data
    print("Running CLIP on VQA V2 data...\n")
    if answer_mode == "answer":
        vqa_results, accuracy = run_CLIP_on_VQA(val_dataloader)
        save_json(vqa_results, f"{results_path}/clip_vqa_results.json")

    elif answer_mode == "no_ans":
        vqa_results, accuracy = test_CLIP_on_VQA(val_dataloader, dataset=val_dataset)

        save_json(vqa_results, f"{results_path}/clip_vqa_results_no_ans.json")

    # Save the evaluation results
    with open(f"{results_path}/evaluation_results_{dataset_type}.txt", "a") as f:
        f.write(f"Mode: CLIP Zero shot Model Evaluation in {answer_mode} \n")
        f.write(f"Dataset: VQA \n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Validation data size: {len(vqa_results)} \n")
        f.write(f"Number of unique answers from all types: {top_k_ans} \n")
        f.write(f"Number of QA pairs: {num_qa_pairs} \n")
        f.write("\n")


if __name__ == "__main__":
    if dataset_type == "vcr":
        vcr_main()

    elif dataset_type == "vqa":
        vqa_main()

    else:
        raise ValueError("Dataset type not recognized")


# python3 -W ignore main.py --annots_dir data/vcr1annots --image_dir data/vcr1images --dataset vcr --ans_mode no_ans
# python3 -W ignore main.py --annots_dir data/vqa_v2 --image_dir data/vqa_v2 --dataset vqa --ans_mode no_ans
