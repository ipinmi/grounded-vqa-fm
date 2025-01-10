import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import subprocess

from data_loading.vcr_dataloader import (
    VCRDataExtractor,
    VCRDataset,
    VCRDataLoader,
    BatchSampler,
)
from data_loading.vqa_dataloader import load_vqa_data, VQADataset

from cross_modal_clip.model import CLIPwithAttention

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
    "--batch_size",
    help="Batch size for training",
    default=64,
    required=False,
    type=int,
)

parser.add_argument(
    "--num_epochs",
    help="Number of epochs to train the model",
    default=20,
    required=False,
    type=int,
)

parser.add_argument(
    "--dataset",
    help="Dataset to be used",
    default="vcr",
    required=True,
)

parser.add_argument(
    "--results_path",
    help="Directory path for saving the model results",
    default="results",
)

args = parser.parse_args()
ANNOTS_DIR = args.annots_dir
IMAGES_DIR = args.image_dir
LEARN_RATE = args.learn_rate
dataset_type = args.dataset
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
results_path = args.results_path

# make results directory
subprocess.run(["mkdir", "-p", results_path])

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_cma_vcr(
    annots_dir, imgs_dir, learn_rate, batchSize=BATCH_SIZE, num_epochs=NUM_EPOCHS
):

    # Load the pre-trained CLIP model
    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    train_max_pairs = 10000
    val_max_pairs = 1000

    # Load the VCR train dataset
    extracted_train_vcr = VCRDataExtractor(
        annots_dir,
        imgs_dir,
        mode="answer",
        split="train",
        only_use_relevant_dets=True,
    )
    train_dataset = VCRDataset(extracted_train_vcr, "vqa", load_all=True)
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
    val_dataset = VCRDataset(extracted_val_vcr, "vqa", load_all=True)
    val_batch_sampler = BatchSampler(val_dataset, batch_size=batchSize)
    val_dataloader = VCRDataLoader(val_dataset, batch_sampler=val_batch_sampler)

    # Initialize the model
    num_choices = 4  # number of possible answers
    current_task = "vqa"
    model = CLIPwithAttention(
        clip_model=clip_model, num_answers=num_choices, drop_out=0.1, task=current_task
    ).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
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
            _, image_paths, questions, choices, labels, _ = batch

            labels_tensor = torch.tensor(labels).to(device)
            labels_tensor = torch.argmax(labels_tensor, dim=0).unsqueeze(
                0
            )  # Shape: (1,)

            # Prepare the text inputs (question and the choices) and image inputs
            question_tokens = clip.tokenize(questions[0]).to(device)  # shape: (1, 77)

            # select the choice of the correct answer for the QA-R task
            correct_choice = choices[labels.index(1)]

            choice_tokens = clip.tokenize(correct_choice).to(device)  # shape:  (1, 77)

            # Assuming one image per batch, open and preprocess it
            image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

            # Forward pass
            if current_task == "vqa":
                output = model(image, question_tokens)
            elif current_task == "vqa-r":
                output = model(image, question_tokens, choice_tokens)
            loss = criterion(output, labels_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy and loss
            epoch_train_loss += loss.item()
            _, predicted = output.max(dim=1)
            epoch_train_total += labels_tensor.size(0)
            epoch_train_correct += (
                (predicted == labels_tensor).sum().item()
            )  # Correct predictions

        # Validation
        model.eval()
        epoch_val_correct = 0
        epoch_val_total = 0
        epoch_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                _, image_paths, questions, choices, labels, _ = batch

                labels_tensor = torch.tensor(labels).to(device)
                labels_tensor = torch.argmax(labels_tensor, dim=0).unsqueeze(
                    0
                )  # Shape: (1,)

                # Prepare the text inputs (question and the possible choices) and image inputs
                question_tokens = clip.tokenize(questions[0]).to(
                    device
                )  # shape: (1, 77)

                # select the choice of the correct answer for the QA-R task
                correct_choice = choices[labels.index(1)]

                choice_tokens = clip.tokenize(correct_choice).to(
                    device
                )  # shape:  (1, 77)

                # Forward pass
                if current_task == "vqa":
                    output = model(image, question_tokens)
                elif current_task == "vqa-r":
                    output = model(image, question_tokens, choice_tokens)

                # Compute the loss
                loss = criterion(output, labels_tensor)

                # Compute the accuracy and loss
                epoch_val_loss += loss.item()
                _, predicted = output.max(dim=1)
                epoch_val_total += labels_tensor.size(0)
                epoch_val_correct += (predicted == labels_tensor).sum().item()

        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        epoch_train_loss = epoch_train_loss / len(train_dataloader)

        epoch_val_accuracy = epoch_val_correct / epoch_val_total
        epoch_val_loss = epoch_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}"
        )
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

        # Update the learning rate
        # scheduler.step()

        # Save the best performing model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                model.state_dict(),
                f"{results_path}/vcr_clip_linear.pt",
            )

    return model


def train_cma_vqa(DATA_DIR, learn_rate, batchSize=BATCH_SIZE, num_epochs=NUM_EPOCHS):

    # Load the pre-trained CLIP model
    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    # Load the VQA train dataset and select the top k answers from each answer type
    # train questions: 443,757 questions
    train_top_k = 10
    val_top_k = 10
    train_max_pairs = 1000
    val_max_pairs = 50

    train_qa_pairs, train_possible_answers_by_type, train_answers = load_vqa_data(
        DATA_DIR,
        split="train",
        top_k=train_top_k,
        max_pairs=train_max_pairs,
        load_all=False,
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
    val_qa_pairs, val_possible_answers_by_type, val_answers = load_vqa_data(
        DATA_DIR, split="val", top_k=val_top_k, max_pairs=val_max_pairs, load_all=False
    )
    val_dataset = VQADataset(
        val_qa_pairs,
        split="val",
        filepath=DATA_DIR,
        answers_by_type=val_possible_answers_by_type,
        all_answers=val_answers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)

    # Initialize the model
    model = CLIPwithAttention(
        clip_model=clip_model, num_answers=train_top_k, drop_out=0.1, task="vqa"
    ).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
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
            # annot_ids = batch["annot_id"].detach().numpy()
            questions = batch["question"]
            # answers = batch["answer"]
            image_paths = batch["image_path"]
            answer_targets = batch["answer_idx"].to(device)

            # Prepare the text inputs (questions) and image inputs
            question_toks = clip.tokenize(questions).to(device)
            images = [Image.open(image_path) for image_path in image_paths]
            image_features = torch.stack([preprocessor(i) for i in images]).to(device)

            # Forward pass
            output = model(image_features, question_toks)
            loss = criterion(output, answer_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy and loss
            epoch_train_loss += loss.item()
            _, predicted = output.max(dim=1)
            epoch_train_total += answer_targets.size(0)
            epoch_train_correct += predicted.eq(answer_targets).sum().item()

        # Validation
        model.eval()
        epoch_val_correct = 0
        epoch_val_total = 0
        epoch_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                # annot_ids = batch["annot_id"].detach().numpy()
                questions = batch["question"]
                # answers = batch["answer"]
                image_paths = batch["image_path"]
                answer_targets = batch["answer_idx"].to(device)

                # Prepare the text inputs (questions) and image inputs
                question_toks = clip.tokenize(questions).to(device)
                images = [Image.open(image_path) for image_path in image_paths]
                image_features = torch.stack([preprocessor(i) for i in images]).to(
                    device
                )

                # Forward pass
                output = model(image_features, question_toks)
                loss = criterion(output, answer_targets)

                # Compute the accuracy and loss
                epoch_val_loss += loss.item()
                _, predicted = output.max(dim=1)
                epoch_val_total += answer_targets.size(0)
                epoch_val_correct += predicted.eq(answer_targets).sum().item()

        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        epoch_train_loss = epoch_train_loss / len(train_dataloader)

        epoch_val_accuracy = epoch_val_correct / epoch_val_total
        epoch_val_loss = epoch_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}"
        )
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

        # Update the learning rate
        scheduler.step()

        # Save the best performing model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                model.state_dict(),
                f"{results_path}/vqa_clip_linear.pt",
            )
    return model


def vqa_usage():
    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    DATA_DIR = "data/vqa_v2"
    batchSize = 4
    train_qa_pairs, train_possible_answers_by_type, train_answers = load_vqa_data(
        DATA_DIR, split="train", top_k=10, max_pairs=100, load_all=False
    )
    train_dataset = VQADataset(
        train_qa_pairs,
        split="train",
        filepath=DATA_DIR,
        answers_by_type=train_possible_answers_by_type,
        all_answers=train_answers,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    model = CLIPwithAttention(clip_model, num_answers=10, drop_out=0.1).to(device)

    for batch in tqdm(train_dataloader):
        # annot_ids = batch["annot_id"].detach().numpy()
        questions = batch["question"]
        # answers = batch["answer"]
        image_paths = batch["image_path"]
        answer_targets = batch["answer_idx"].to(device)

        # Prepare the text inputs (questions) and image inputs
        question_toks = clip.tokenize(questions).to(device)
        images = [Image.open(image_path) for image_path in image_paths]
        image_features = torch.stack([preprocessor(i) for i in images]).to(device)

        print(f"Question Tokens Shape: {question_toks.shape}")
        print(f"Image Features Shape: {image_features.shape}")

        output = model(image_features, question_toks)

        print(f"Output Shape: {output.shape}")


if __name__ == "__main__":
    if dataset_type == "vcr":
        linear_clip_model = train_cma_vcr(ANNOTS_DIR, IMAGES_DIR, LEARN_RATE)
    elif dataset_type == "vqa":
        linear_clip_model = train_cma_vqa(ANNOTS_DIR, LEARN_RATE)
    else:
        raise ValueError("Dataset type not recognized")

    # Test the model
    # vqa_usage()

# Sample usage: python3 cma_train.py --annots_dir data/vcr1annots --image_dir data/vcr1images --learn_rate 0.001 --batch_size 4 --num_epochs 2 --dataset vcr
# Sample usage: python3 cma_train.py --annots_dir data/vqa_v2 --image_dir data/vqa_v2 --learn_rate 0.001 --batch_size 4 --num_epochs 2 --dataset vqa
