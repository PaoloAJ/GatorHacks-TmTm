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
    Uses lazy loading to avoid blocking container startup.
    """

    _instance = None
    _encoder = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one encoder instance"""
        if cls._instance is None:
            cls._instance = super(EncoderService, cls).__new__(cls)
        return cls._instance

    def _ensure_initialized(self):
        """Lazy initialization - only load model when first needed"""
        if not self._initialized:
            print("🔄 Initializing CNN encoder (lazy loading)...")
            self._encoder = CNNEncoder(
                embedding_dim=settings.embedding_dim,
                use_pretrained=settings.use_pretrained
            )

            # Load fine-tuned weights if available
            from pathlib import Path
            finetuned_path = Path(__file__).parent.parent / "models" / "finetuned_encoder.pth"
            if finetuned_path.exists():
                print(f"✅ Loading fine-tuned weights from {finetuned_path}")
                self._encoder.load_weights(str(finetuned_path))
            else:
                print("⚠️  Using pre-trained ImageNet weights (no fine-tuned model found)")

            self._initialized = True
            print("✅ CNN encoder initialized successfully")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single image to an embedding vector.

        Args:
            image: PIL Image object (RGB)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        self._ensure_initialized()
        return self._encoder.encode(image)

    def encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode multiple images in a batch.

        Args:
            images: List of PIL Image objects

        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        self._ensure_initialized()
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
        self._ensure_initialized()
        return self._encoder.embedding_dim


# Global service instance
encoder_service = EncoderService()
