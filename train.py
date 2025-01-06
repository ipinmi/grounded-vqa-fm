import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from data_loading.vcr_dataloader import (
    VCRDataExtractor,
    VCRDataset,
    VCRDataLoader,
    BatchSampler,
)
from data_loading.vqa_dataloader import load_vqa_data, VQADataset
from extended_clip.linear_clip import VCRLinearModel, VQALinearModel

# Arguments for data preprocessing and loading
parser = argparse.ArgumentParser()
parser.add_argument(
    "--annots_dir",
    help="Directory path for annotations where val.jsonl, val.jsonl, test.jsonl are stored",
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
    "--learn_rate",
    help="Learning rate for the model",
    default=1e-4,
    required=True,
    type=float,
)

parser.add_argument(
    "--dataset",
    help="Dataset to be used",
    default="vcr",
    required=True,
)

args = parser.parse_args()
ANNOTS_DIR = args.annots_dir
IMAGES_DIR = args.image_dir
LEARN_RATE = args.learn_rate
dataset_type = args.dataset


def train_linear_vcr(annots_dir, imgs_dir, learn_rate, batchSize=4, num_epochs=20):
    # Load the pre-trained CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    # Load the VCR train dataset
    extracted_train_vcr = VCRDataExtractor(
        annots_dir,
        imgs_dir,
        mode="answer",
        split="train",
        only_use_relevant_dets=True,
    )
    train_dataset = VCRDataset(extracted_train_vcr, "vqa", size=1000)
    train_batch_sampler = BatchSampler(train_dataset, batch_size=batchSize)
    train_dataloader = VCRDataLoader(train_dataset, batch_sampler=train_batch_sampler)

    # Load the VCR Validation dataset
    extracted_val_vcr = VCRDataExtractor(
        annots_dir,
        imgs_dir,
        mode="answer",
        split="val",
        only_use_relevant_dets=True,
    )
    val_dataset = VCRDataset(extracted_val_vcr, "vqa", size=500)
    val_batch_sampler = BatchSampler(val_dataset, batch_size=batchSize)
    val_dataloader = VCRDataLoader(val_dataset, batch_sampler=val_batch_sampler)

    # Define the VCR linear model
    num_choices = 4  # number of possible answers
    model = VCRLinearModel(clip_model, num_choices)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initialize the best validation loss
    best_val_loss = float("inf")

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch in tqdm(train_dataloader):
            annot_id, image_paths, questions, choices, labels, detections = batch

            labels = torch.tensor(labels).to(device)

            # Prepare the text inputs (question and the possible choices) and image inputs
            question_tokens = clip.tokenize(questions[0]).to(device)  # shape: (1, 77)
            choices_tokens = clip.tokenize(choices).to(
                device
            )  # shape:  (num_choices, 77)

            # Assuming one image per batch, open and preprocess it
            image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

            # Forward pass
            outputs = model(image, question_tokens, choices_tokens)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy and loss
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            epoch_train_total += 1
            epoch_train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        epoch_val_correct = 0
        epoch_val_total = 0
        epoch_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                annot_id, image_paths, questions, choices, labels, detections = batch

                labels = torch.tensor(labels).to(device)

                # Prepare the text inputs (question and the possible choices) and image inputs
                question_tokens = clip.tokenize(questions[0]).to(
                    device
                )  # shape: (1, 77)
                choices_tokens = clip.tokenize(choices).to(
                    device
                )  # shape:  (num_choices, 77)

                # Assuming one image per batch, open and preprocess it
                image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

                # Forward pass
                outputs = model(image, question_tokens, choices_tokens)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute the accuracy and loss
                epoch_train_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                epoch_train_total += 1
                epoch_train_correct += predicted.eq(labels).sum().item()

        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        epoch_val_accuracy = epoch_val_correct / epoch_val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {epoch_train_loss / len(train_dataloader):.4f}, Train Accuracy: {epoch_train_accuracy:.4f}"
        )
        print(
            f"Val Loss: {epoch_val_loss / len(val_dataloader):.4f}, Val Accuracy: {epoch_val_accuracy:.4f}"
        )

        scheduler.step()

        # Save the best performing model
        """if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                model.state_dict(), f"results/vqa_clip_linear_{epoch_val_loss:.2f}.pt"
            )"""
    return model


def train_linear_vqa(DATA_DIR, LEARN_RATE, batchSize=32, num_epochs=20):
    # Load the pre-trained CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    # Load the VQA train dataset and select the top k answers from each answer type
    # train questions: 443,757 questions

    train_qa_pairs, train_possible_answers_by_type, train_answers, _ = load_vqa_data(
        DATA_DIR, split="train", top_k=100, max_pairs=100000
    )
    train_dataset = VQADataset(
        train_qa_pairs,
        split="train",
        filepath=DATA_DIR,
        answers_by_type=train_possible_answers_by_type,
        all_answers=train_answers,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    # Validation dataset
    # val questions: 214,354 questions
    val_qa_pairs, val_possible_answers_by_type, val_answers, _ = load_vqa_data(
        DATA_DIR, split="val", top_k=50, max_pairs=10000
    )
    val_dataset = VQADataset(
        val_qa_pairs,
        split="val",
        filepath=DATA_DIR,
        answers_by_type=val_possible_answers_by_type,
        all_answers=val_answers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)

    # Define the VQA linear model
    num_answers = len(train_answers)
    model = VQALinearModel(clip_model, num_answers)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initialize the best validation loss
    best_val_loss = float("inf")

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        for batch in tqdm(train_dataloader):
            annot_ids = batch["annot_id"].detach().numpy()
            questions = batch["question"]
            answers = batch["answer"]
            image_paths = batch["image_path"]
            answer_targets = batch["answer_idx"].to(device)

            # Prepare the text inputs (questions) and image inputs
            question_toks = clip.tokenize(questions).to(device)
            images = [Image.open(image_path) for image_path in image_paths]
            image_features = torch.stack([preprocessor(i) for i in images]).to(device)

            # Forward pass
            outputs = model(question_toks, image_features)
            loss = criterion(outputs, answer_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy and loss
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            epoch_train_total += answer_targets.size(0)
            epoch_train_correct += predicted.eq(answer_targets).sum().item()

        # Validation
        model.eval()
        epoch_val_correct = 0
        epoch_val_total = 0
        epoch_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                annot_ids = batch["annot_id"].detach().numpy()
                questions = batch["question"]
                answers = batch["answer"]
                image_paths = batch["image_path"]
                answer_targets = batch["answer_idx"].to(device)

                # Prepare the text inputs (questions) and image inputs
                question_toks = clip.tokenize(questions).to(device)
                images = [Image.open(image_path) for image_path in image_paths]
                image_features = torch.stack([preprocessor(i) for i in images]).to(
                    device
                )

                # Forward pass only
                outputs = model(question_toks, image_features)
                loss = criterion(outputs, answer_targets)

                # Compute the accuracy and loss
                epoch_val_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                epoch_val_total += answer_targets.size(0)
                epoch_val_correct += predicted.eq(answer_targets).sum().item()

        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        epoch_val_accuracy = epoch_val_correct / epoch_val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {epoch_train_loss / len(train_dataloader):.4f}, Train Accuracy: {epoch_train_accuracy:.4f}"
        )
        print(
            f"Val Loss: {epoch_val_loss / len(val_dataloader):.4f}, Val Accuracy: {epoch_val_accuracy:.4f}"
        )

        scheduler.step()

        # Save the best performing model
        """if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                model.state_dict(), f"results/vqa_clip_linear_{epoch_val_loss:.2f}.pt"
            )"""
    return model


if __name__ == "__main__":
    if dataset_type == "vcr":
        linear_clip_model = train_linear_vcr(ANNOTS_DIR, IMAGES_DIR, LEARN_RATE)
    elif dataset_type == "vqa":
        linear_clip_model = train_linear_vqa(ANNOTS_DIR, LEARN_RATE)
    else:
        raise ValueError("Dataset type not recognized")


# Sample usage: python3 train.py --annots_dir data/vcr1annots --image_dir data/vcr1images --learn_rate 0.001 --dataset vcr
# Sample usage: python3 train.py --data_dir data/vqa_v2 --learn_rate 0.001 --dataset vqa
