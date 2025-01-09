import torch
import torch.nn as nn
import clip
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from cross_modal_clip.cross_modal_fusion import CrossModalFusion

######

from data_loading.vqa_dataloader import load_vqa_data, VQADataset

##########

CLIP_DIM = 512
HIDDEN_DIM = 256
NUM_HEADS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPwithAttention(nn.Module):
    def __init__(self, clip_model, num_answers: int, drop_out: float, **kwargs):
        super(CLIPwithAttention, self).__init__()

        self.task = kwargs.get("task", "vqa")

        # Load the pre-trained CLIP model
        self.clip_model = clip_model

        # Freeze the CLIP model during training
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.cross_modal_attention = CrossModalFusion(
            image_dim=CLIP_DIM,
            text_dim=CLIP_DIM,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
        )

        # define the linear layer for the VQA classification task
        if self.task == "vqa":
            self.classifier = nn.Sequential(
                nn.Linear(
                    HIDDEN_DIM,
                    HIDDEN_DIM,
                ),
                nn.ReLU(),
                nn.LayerNorm(HIDDEN_DIM),
                nn.Dropout(drop_out),
                nn.Linear(HIDDEN_DIM, num_answers),
            ).to(device)

        elif self.task == "vqa-r":
            self.classifier = nn.Sequential(
                nn.Linear(
                    HIDDEN_DIM,
                    HIDDEN_DIM,
                ),
                nn.ReLU(),
                nn.LayerNorm(HIDDEN_DIM),
                nn.Dropout(drop_out),
                nn.Linear(HIDDEN_DIM, num_answers),
            ).to(device)

            self.joint_fusion = nn.Sequential(
                nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(drop_out),
            ).to(device)

    def extract_clip_features(self, images, texts, **kwargs):

        choices = kwargs.get("answers", None)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(texts)

            # convert to float32 for compatibility
            image_features = image_features.float()
            text_features = text_features.float()

            # Normalize the image and text features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            if self.task == "vqa-r":
                choices_features = self.clip_model.encode_text(choices)
                choices_features = choices_features.float()
                choices_features /= choices_features.norm(dim=-1, keepdim=True)

                return image_features, text_features, choices_features

            elif self.task == "vqa":
                return image_features, text_features

    def forward(self, images, questions, **kwargs):

        choices = kwargs.get("choices", None)

        if self.task == "vqa":
            image_features, question_features = self.extract_clip_features(
                images, questions
            )

            # Perform cross-modal attention over the image and question features
            fused_features = self.cross_modal_attention(
                image_features, question_features
            )  # Shape: (batch_size, hidden_dim)

            # Perform classification
            output = self.classifier(fused_features)  # Shape: (batch_size, num_answers)

        elif self.task == "vqa-r":
            image_features, question_features, choices_features = (
                self.extract_clip_features(images, questions, choices=choices)
            )

            # Perform cross-modal attention over the image and question features
            fused_questions = self.cross_modal_attention(
                image_features, question_features
            )

            # Perform cross-modal attention over the image and choices features
            fused_choices = self.cross_modal_attention(image_features, choices_features)

            # Concatenate the fused question and choices features
            joint_features = torch.cat(
                [fused_questions, fused_choices], dim=1
            )  # Shape: (batch_size, hidden_dim * 2)

            # Perform joint fusion
            joint_features = self.joint_fusion(
                joint_features
            )  # Shape: (batch_size, hidden_dim)

            # Perform classification
            output = self.classifier(joint_features)  # Shape: (batch_size, num_answers)
        return output
