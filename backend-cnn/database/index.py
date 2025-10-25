from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Dict, Optional, Tuple


class ArtDatabase:

    # Manages a Pinecone vector database for artwork similarity search

    def __init__(
      self,
      api_key: str,
      index_name: str = "art-imbeddings",
      dimension: int = 512,
      metric: str = "cosine",
      cloud: str = "aws",
      region: str = "us-east-1"
      ):
        
        
        # Initialize Pinecone database connection.
        
        # Args:
        #     api_key: Your Pinecone API key
        #     index_name: Name for your Pinecone index
        #     dimension: Dimension of embeddings (must match your encoder)
        #     metric: Distance metric ('cosine', 'euclidean', or 'dotproduct')
        #     cloud: Cloud provider ('aws', 'gcp', or 'azure')
        #     region: Cloud region

        self.index_name = index_name
        self.dimension = dimension

        # Initialize Pinecone
        self.pc = Pinecone(Pinecone(api_key=api_key))

        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )

        self.index = self.pc.Index(index_name)

    def add_artwork(
        self,
        artwork_id: str,
        embedding: np.ndarray,
        metadata: Dict
    ):
        
    # Add a single artwork to the database.
        
    #     Args:
    #         artwork_id: Unique identifier for the artwork
    #         embedding: Vector embedding (numpy array of shape (dimension,))
    #         metadata: Dictionary with artwork information
    #                  e.g., {'artist': 'Van Gogh', 'title': 'Starry Night', 
    #                         'year': 1889, 'style': 'Post-Impressionism'}

        # Convert numpy array to list
        vector = embedding.tolist()

        #Upsert to Pinecone
        self.index.upsert(
            vectors=[
                {
                    "id": artwork_id,
                    "values": vector,
                    "metadata": metadata
                }
            ]
        )

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_metadata: bool = True
    ):
        
