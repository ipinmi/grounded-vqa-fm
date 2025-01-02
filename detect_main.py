import argparse
import matplotlib.pyplot as plt
import json


##MODULES
from baseline_vqa.clip_interface import run_CLIP_on_VCR
from data_loading.vcr_dataloader import VCRDataExtractor, VCRDataset, VCRDataLoader
from clip_detector.object_detector import *

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

args = parser.parse_args()
VCR_ANNOTS_DIR = args.annots_dir
VCR_IMAGES_DIR = args.image_dir


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    extracted_vcr = VCRDataExtractor(VCR_ANNOTS_DIR, VCR_IMAGES_DIR)
    dataset = VCRDataset(extracted_vcr)
    dataloader = VCRDataLoader(dataset, batch_size=4)

    # Run the CLIP model
    # vqa_results = run_CLIP(dataloader)
    # save_json(results, "results/clip_vqa_results.json")

    # Run the CLIP detector
    iou_results = {}
    accuracy_results = []

    for idx, batch in enumerate(dataloader):
        if idx < 10:
            image_paths, questions, answers, labels, detection_metadata = batch

            object_cats = detection_metadata["objects_cats"][0]
            object_bbox = detection_metadata["boxes"]

            # prompts = generate_prompts(object_cats)

            boxes = run_detection(
                image_paths[0],
                object_cats,
                patch_size=64,
                window=4,
                stride=1,
                decision_threshold=0.3,
            )

            # caluclate the IOU over each detected object
            object_iou = {}
            ground_truth = object_bbox[0].numpy()
            for i, obj in enumerate(boxes.keys()):
                # print(f"Processing object: {obj}")
                ground_truth_box = ground_truth[i]
                predicted_box = boxes[obj][0]

                # print(f"Ground truth: {ground_truth_box}")
                # print(f"Predicted box: {predicted_box}")

                iou = calculate_iou(ground_truth_box, predicted_box)

                object_iou[obj] = iou

                if iou > 0.5:
                    accuracy_results.append(1)
                else:
                    accuracy_results.append(0)

            iou_results[image_paths[0]] = object_iou

    sum_accuracy = sum(accuracy_results)
    if sum_accuracy == 0:
        total_accuracy = 0.0
    else:
        total_accuracy = sum(accuracy_results) / len(accuracy_results)

    print(f"Accuracy over {len(accuracy_results)} {total_accuracy}")


if __name__ == "__main__":
    main()


# python3 main.py -W ignore --annots_dir data/vcr1annots --image_dir data/vcr1images
