import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import subprocess
import sys
import matplotlib.pyplot as plt

from data_loading.vcr_dataloader import (
    VCRDataExtractor,
    VCRDataset,
    VCRDataLoader,
    BatchSampler,
)
from data_loading.vqa_dataloader import load_vqa_data, VQADataset
from linear_clip.linear_clip import VCRLinearModel, VQALinearModel

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


def train_linear_vcr(
    annots_dir, imgs_dir, learn_rate, batchSize=BATCH_SIZE, num_epochs=NUM_EPOCHS
):
    """
    Train the linear model for the VCR dataset using the image and text features from the frozen CLIP model.

    Args:
    annots_dir: str, path to the directory containing the VCR annotations
    imgs_dir: str, path to the directory containing the VCR images
    learn_rate: float, learning rate for the model
    batchSize: int, batch size for training (default: 4)
    num_epochs: int, number of epochs to train the model (default: 10)


    Returns:
    model: VCRLinearModel, trained linear model for the VCR dataset
    Also plots the evaluation metrics (loss and accuracy) for the training and validation sets.
    """

    # Load the pre-trained CLIP model
    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    # Load the VCR train dataset
    train_max_pairs = 100000
    val_max_pairs = 10000

    extracted_train_vcr = VCRDataExtractor(
        annots_dir,
        imgs_dir,
        mode="answer",
        split="train",
        only_use_relevant_dets=True,
        load_all=False,
        size=train_max_pairs,
    )
    train_dataset = VCRDataset(extracted_train_vcr, "vqa")
    train_batch_sampler = BatchSampler(train_dataset, batch_size=batchSize)
    train_dataloader = VCRDataLoader(train_dataset, batch_sampler=train_batch_sampler)

    # Load the VCR Validation dataset
    extracted_val_vcr = VCRDataExtractor(
        annots_dir,
        imgs_dir,
        mode="answer",
        split="val",
        only_use_relevant_dets=True,
        load_all=False,
        size=val_max_pairs,
    )
    val_dataset = VCRDataset(extracted_val_vcr, "vqa")
    val_batch_sampler = BatchSampler(val_dataset, batch_size=batchSize)
    val_dataloader = VCRDataLoader(val_dataset, batch_sampler=val_batch_sampler)

    # Define the VCR linear model
    num_choices = 4  # number of possible answers
    model = VCRLinearModel(clip_model, num_choices, task="vqa")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initialize the best validation loss
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    train_acc = []
    val_acc = []

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

            # Prepare the text inputs (question and the possible choices) and image inputs
            question_tokens = clip.tokenize(questions[0]).to(device)  # shape: (1, 77)

            # select the choice of the correct answer for the QA-R task
            correct_choice = choices[labels.index(1)]

            choice_tokens = clip.tokenize(correct_choice).to(device)  # shape:  (1, 77)

            # Assuming one image per batch, open and preprocess it
            image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

            # Forward pass
            outputs = model(image, question_tokens, choice_tokens)
            loss = criterion(outputs, labels_tensor)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy and loss
            epoch_train_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            epoch_train_total += labels_tensor.size(0)
            epoch_train_correct += (predicted == labels_tensor).sum().item()

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

                # Assuming one image per batch, open and preprocess it
                image = preprocessor(Image.open(image_paths[0])).unsqueeze(0).to(device)

                # Forward pass
                outputs = model(image, question_tokens, choice_tokens)
                loss = criterion(outputs, labels_tensor)

                # Compute the accuracy and loss
                epoch_val_loss += loss.item()
                _, predicted = outputs.max(dim=1)
                epoch_val_total += labels_tensor.size(0)
                epoch_val_correct += (predicted == labels_tensor).sum().item()

        epoch_train_accuracy = epoch_train_correct / epoch_train_total
        epoch_val_accuracy = epoch_val_correct / epoch_val_total

        epoch_train_loss /= len(train_dataloader)
        epoch_val_loss /= len(val_dataloader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        train_acc.append(epoch_train_accuracy)
        val_acc.append(epoch_val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {epoch_train_loss / len(train_dataloader):.4f}, Train Accuracy: {epoch_train_accuracy:.4f}"
        )
        print(
            f"Val Loss: {epoch_val_loss / len(val_dataloader):.4f}, Val Accuracy: {epoch_val_accuracy:.4f}"
        )
        sys.stdout.flush()

        # scheduler.step()

        # Save the best performing model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(
                model.state_dict(),
                f"{results_path}/vcr_clip_linear.pt",
            )

    # Plot the evaluation metrics
    run_and_plot(train_losses, val_losses, train_acc, val_acc, num_epochs)
    return model


def train_linear_vqa(DATA_DIR, LEARN_RATE, batchSize=BATCH_SIZE, num_epochs=NUM_EPOCHS):
    """
    Train the linear model for the VQA dataset using the image and text features from the frozen CLIP model.

    Args:
    DATA_DIR: str, path to the directory containing the VQA annotations
    LEARN_RATE: float, learning rate for the model
    batchSize: int, batch size for training
    num_epochs: int, number of epochs to train the model

    Returns:
    model: VQALinearModel, trained linear model for the VQA dataset
    Also plots the evaluation metrics (loss and accuracy) for the training and validation sets.
    """

    # Load the pre-trained CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    # Load the VQA train dataset and select the top k answers from each answer type
    # train questions: 443,757 questions

    train_qa_pairs, train_possible_answers_by_type, train_answers = load_vqa_data(
        DATA_DIR, split="train", top_k=100, max_pairs=100000, load_all=False
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
        DATA_DIR, split="val", top_k=100, max_pairs=10000, load_all=False
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
    train_losses = []
    val_losses = []

    train_acc = []
    val_acc = []

    sys.stdout.flush()

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

        epoch_train_loss /= len(train_dataloader)
        epoch_val_loss /= len(val_dataloader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        train_acc.append(epoch_train_accuracy)
        val_acc.append(epoch_val_accuracy)

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
                f"{results_path}/vqa_clip_linear.pt",
            )

    # Plot the evaluation metrics
    run_and_plot(train_losses, val_losses, train_acc, val_acc, num_epochs)

    return model


def run_and_plot(
    train_losses,
    val_losses,
    train_acc_scores,
    val_acc_scores,
    n_epochs,
):
    """
    This function collects the evaluation results and plots the evaluation metrics.
    The evaluation results are saved to a text file, and the plots are saved as a PNG file.
    """
    # Ensure the data lengths match the epochs
    train_losses = train_losses[:n_epochs]
    train_acc_scores = train_acc_scores[:n_epochs]

    ### Save train, val, test losses and acc scores for combined plotting with ensemble models
    with open(f"{results_path}/cma_{dataset_type}_loss_acc.txt", "a") as f:
        f.write(f"Model: Cross Modal CLIP Model Evaluation\n")
        f.write("\n")
        f.write(f"Train Loss: {train_losses}\n")
        f.write(f"Val Loss: {val_losses}\n")
        f.write("\n")
        f.write(f"Train acc Score: {train_acc_scores}\n")
        f.write(f"Val acc Score: {val_acc_scores}\n")
        f.write("\n")

    # Save the evaluation results
    with open(f"{results_path}/cma_{dataset_type}_results.txt", "a") as f:
        f.write(f"Model: Cross Modal CLIP Model Evaluation\n")
        f.write("\n")
        f.write(f"Train Loss: {round(train_losses[-1], 3)}\n")
        f.write(f"Val Loss: {round(val_losses[-1], 3)}\n")
        f.write("\n")
        f.write(f"Train acc Score: {round(train_acc_scores[-1], 3)}\n")
        f.write(f"Val acc Score: {round(val_acc_scores[-1], 3)}\n")
        f.write("\n")

    # Plot the losses with epochs on the x-axis
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, n_epochs + 1),
        train_losses,
        label="Train Loss",
        color="blue",
        linestyle="-",
    )
    plt.plot(
        range(1, n_epochs + 1),
        val_losses,
        label="Validation Loss",
        color="red",
        linestyle=":",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross Entropy Loss per Epoch (Train, Val)")
    plt.grid(True)
    plt.legend()

    # Plot the acc scores with epochs on the x-axis
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, n_epochs + 1),
        train_acc_scores,
        label="Train Accuracy",
        color="green",
        linestyle="-",
    )
    plt.plot(
        range(1, n_epochs + 1),
        val_acc_scores,
        label="Validation Accuracy",
        color="orange",
        linestyle=":",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Score")
    plt.title("Accuracy Score per Epoch (Train, Val)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{results_path}/{dataset_type}_evaluation_plot.png")
    plt.show()


if __name__ == "__main__":
    if dataset_type == "vcr":
        linear_clip_model = train_linear_vcr(
            ANNOTS_DIR, IMAGES_DIR, LEARN_RATE, BATCH_SIZE, NUM_EPOCHS
        )
    elif dataset_type == "vqa":
        linear_clip_model = train_linear_vqa(
            ANNOTS_DIR, LEARN_RATE, BATCH_SIZE, NUM_EPOCHS
        )
    else:
        raise ValueError("Dataset type not recognized")


# Sample usage: python3 linear_train.py --annots_dir data/vcr1annots --image_dir data/vcr1images --learn_rate 0.001 --batch_size 4 --num_epochs 20 --dataset vcr
# Sample usage: python3 linear_train.py --annots_dir data/vqa_v2 --image_dir data/vqa_v2 --learn_rate 1e-3  --batch_size 64 --num_epochs 20 --dataset vqa
