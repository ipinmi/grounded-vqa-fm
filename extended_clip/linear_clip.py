import torch
import torch.nn as nn


class VCRLinearModel(nn.Module):
    """
    Defines the linear model for the VCR classification task using the CLIP model.
    The model takes an image, a question, and the mutliple choice answers as input
    and outputs the probabilities of the choices.
    """

    def __init__(self, clip_model, num_choices, hidden_size=512, drop_out=0.5):
        super(VCRLinearModel, self).__init__()
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
            nn.Linear(hidden_size, num_choices),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images, questions, choices):
        # Extract the image and text (question and the possible choices) features from the frozen CLIP model
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            question_features = self.clip_model.encode_text(questions)
            choices_features = self.clip_model.encode_text(choices)

            # convert to float32 for compatibility with the classifier
            image_features = image_features.float()
            question_features = question_features.float()
            choices_features = choices_features.float()

        # Normalize the image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        question_features /= question_features.norm(dim=-1, keepdim=True)
        choices_features /= choices_features.norm(dim=-1, keepdim=True)

        # Create the joint image-text embedding
        joint_features = torch.cat(
            [image_features, question_features, choices_features], dim=1
        )

        # Pass the joint features through the classifier
        logits = self.classifier(joint_features)

        # Softmax to get probabilities
        output = self.softmax(logits)

        return output


class VQALinearModel(nn.Module):
    """
    Defines the linear model for the VQA classification task using the CLIP model.
    The model takes an image and a question as input and
    outputs the probabilities over the selected number of answers from each answer type.
    """

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
            question_features = self.clip_model.encode_text(questions)

            # convert to float32 for compatibility with the classifier
            image_features = image_features.float()
            question_features = question_features.float()

        # Normalize the image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        question_features /= question_features.norm(dim=-1, keepdim=True)

        # Create the joint image-text embedding
        joint_features = torch.cat([image_features, question_features], dim=1)

        # Pass the joint features through the classifier
        output = self.classifier(joint_features)

        return output
