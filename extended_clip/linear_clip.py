import torch
import clip
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import argparse

from data_loading.vqa_dataloader import load_vqa_data, VQADataset

# Arguments for data preprocessing and loading
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    help="Directory path for annotations where train.jsonl, val.jsonl, test.jsonl are stored",
    default="data/vcr1annots",
    required=True,
)

parser.add_argument(
    "--learn_rate",
    help="Learning rate for the model",
    default=1e-4,
    required=True,
    type=float,
)

args = parser.parse_args()
DATA_DIR = args.data_dir
LEARN_RATE = args.learn_rate


class VQALinearModel(nn.Module):
    def __init__(self, clip_model, num_answers, hidden_size=512, drop_out=0.5):
        super(VQALinearModel, self).__init__()
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.clip_model = clip_model

        # Freeze the CLIP model during training
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # define the linear layer for the VQA classification task
        self.classifier = nn.Sequential(
            nn.Linear(
                clip_model.visual.output_dim + clip_model.text_projection.shape[1],
                hidden_size,
            ),  # both of dimension 512
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(hidden_size, num_answers),
        )

    def forward(self, questions, images):
        # Extract the image and text features from the frozen CLIP model
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(questions)

            # convert to float32 for compatibility with the classifier
            image_features = image_features.float()
            text_features = text_features.float()

        # Normalize the image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Create the joint image-text embedding
        joint_features = torch.cat([image_features, text_features], dim=1)

        # Pass the joint features through the classifier
        output = self.classifier(joint_features)

        return output


def train_linear(DATA_DIR, LEARN_RATE, batchSize=32, num_epochs=20):
    # Load the pre-trained CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocessor = clip.load("ViT-B/32", device=device)

    # Load the VQA train dataset and select the top 1500 answers from each answer type
    train_qa_pairs, train_possible_answers_by_type, train_answers = load_vqa_data(
        DATA_DIR, split="train", top_k=1500, max_pairs=10000
    )
    train_dataset = VQADataset(
        train_qa_pairs,
        split="train",
        filepath=DATA_DIR,
        answers_by_type=train_possible_answers_by_type,
        all_answers=train_answers,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

    # Define the VQA linear model
    num_answers = len(train_answers)
    model = VQALinearModel(clip_model, num_answers)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
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
            total_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            train_total += answer_targets.size(0)
            train_correct += predicted.eq(answer_targets).sum().item()

        train_accuracy = train_correct / train_total
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    return model


if __name__ == "__main__":
    linear_clip_model = train_linear(DATA_DIR, LEARN_RATE)

# Sample usage: python3 extended_clip/linear_clip.py --data_dir data/vqa_v2 --learn_rate 0.001
