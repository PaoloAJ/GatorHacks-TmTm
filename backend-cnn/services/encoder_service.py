"""
Encoder service - Abstracts access to the CNN encoder
This prevents direct access to the encoder from routes
"""
import numpy as np
from PIL import Image
from typing import List

from models.cnn_encoder import CNNEncoder
from config import settings


class EncoderService:
    """
    Service class for handling image encoding operations.
    Provides a clean interface to the CNN encoder without exposing internals.
    """

    _instance = None
    _encoder = None

    def __new__(cls):
        """Singleton pattern to ensure only one encoder instance"""
        if cls._instance is None:
            cls._instance = super(EncoderService, cls).__new__(cls)
            cls._encoder = CNNEncoder(
                embedding_dim=settings.embedding_dim,
                use_pretrained=settings.use_pretrained
            )
        return cls._instance

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single image to an embedding vector.

        Args:
            image: PIL Image object (RGB)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self._encoder.encode(image)

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode multiple images in a batch.

        Args:
            images: List of PIL Image objects

        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        return self._encoder.encode_batch(images)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self._encoder.embedding_dim


# Global service instance
encoder_service = EncoderService()
