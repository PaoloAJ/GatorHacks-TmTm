"""
Pinecone service for vector similarity search
"""
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Dict, Optional
import time

from config import settings


class PineconeService:
    """
    Service for managing Pinecone vector database operations.
    Handles embedding storage and similarity search.
    """

    def __init__(self):
        """Initialize Pinecone connection"""
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._index = None

    @property
    def index(self):
        """Lazy load the index"""
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def create_index(self, dimension: int = 512):
        """
        Create a new Pinecone index if it doesn't exist.

        Args:
            dimension: Embedding dimension (default 512)
        """
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',  # Cosine similarity
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Free tier region
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print(f"Index '{self.index_name}' created successfully!")
        else:
            print(f"Index '{self.index_name}' already exists.")

    def upsert_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: List[Dict],
        batch_size: int = 100
    ):
        """
        Upload embeddings to Pinecone in batches.

        Args:
            embeddings: Array of shape (n, embedding_dim)
            ids: List of unique IDs for each embedding
            metadata: List of metadata dicts (artist_name, image_path, etc.)
            batch_size: Number of vectors to upload at once
        """
        vectors = []
        for i, (embedding, vector_id, meta) in enumerate(zip(embeddings, ids, metadata)):
            vectors.append({
                'id': vector_id,
                'values': embedding.tolist(),
                'metadata': meta
            })

            # Upload in batches
            if len(vectors) >= batch_size or i == len(embeddings) - 1:
                self.index.upsert(vectors=vectors)
                vectors = []
                print(f"Uploaded {i + 1}/{len(embeddings)} embeddings")

    def query_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Find most similar vectors in Pinecone.

        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {'artist': 'Van Gogh'})

        Returns:
            List of dicts with 'id', 'score', and 'metadata'
        """
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )

        return [
            {
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]

    def delete_all(self):
        """Delete all vectors from the index (use with caution!)"""
        self.index.delete(delete_all=True)
        print(f"Deleted all vectors from index '{self.index_name}'")

    def get_index_stats(self):
        """Get statistics about the index"""
        return self.index.describe_index_stats()


# Global service instance
pinecone_service = PineconeService()
