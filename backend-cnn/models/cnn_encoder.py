import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
from typing import Optional


class CNNEncoder:
    """
    CNN-based image encoder for style similarity analysis.
    Uses a pre-trained ResNet50 as the backbone and extracts feature embeddings.
    """

    def __init__(self, embedding_dim: int = 512, use_pretrained: bool = True):
        """
        Initialize the CNN encoder.

        Args:
            embedding_dim: Dimension of the output embedding vector
            use_pretrained: Whether to use pre-trained ImageNet weights
        """
        self.embedding_dim = embedding_dim
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load pre-trained ResNet50 and modify for feature extraction
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet50(weights=None)

        # Remove the final classification layer
        # ResNet50 has 2048 features before the FC layer
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        # Add a projection head to get desired embedding dimension
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Move model to device and set to eval mode
        self.feature_extractor.to(self.device)
        self.projection_head.to(self.device)
        self.feature_extractor.eval()
        self.projection_head.eval()

        # Define image preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def encode(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image into a feature vector.

        Args:
            image: PIL Image object (RGB)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        with torch.no_grad():
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract features
            features = self.feature_extractor(img_tensor)

            # Project to embedding space
            embedding = self.projection_head(features)

            # Normalize embedding (L2 normalization for cosine similarity)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)

            # Convert to numpy and return
            return embedding.cpu().numpy().squeeze()

    def encode_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Encode a batch of images.

        Args:
            images: List of PIL Image objects

        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        with torch.no_grad():
            # Preprocess all images
            img_tensors = torch.stack([self.transform(img) for img in images])
            img_tensors = img_tensors.to(self.device)

            # Extract features
            features = self.feature_extractor(img_tensors)

            # Project to embedding space
            embeddings = self.projection_head(features)

            # Normalize embeddings
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy()

    def save_weights(self, path: str):
        """Save the projection head weights."""
        torch.save(
            {
                "projection_head": self.projection_head.state_dict(),
                "embedding_dim": self.embedding_dim,
            },
            path,
        )

    def load_weights(self, path: str):
        """Load the projection head weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.projection_head.load_state_dict(checkpoint["projection_head"])
        self.projection_head.eval()


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for style encoding (alternative to ResNet).
    Lighter weight option if you want to train from scratch.
    """

    def __init__(self, embedding_dim: int = 512):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # L2 normalize
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
