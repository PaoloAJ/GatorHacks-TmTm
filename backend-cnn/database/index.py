from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Dict


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
        self.pc = Pinecone(api_key=api_key)

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

    def add_artworks_batch(
        self,
        artwork_ids: List[str],
        embeddings: np.ndarray,
        metadata_list: List[Dict],
        batch_size: int = 10000
    ):
        
        # Add multiple artworks in batches
        # Args:
        #     artwork_ids: List of unique IDs
        #     embeddings: Numpy array of shape (n_artworks, dimension)
        #     metadata_list: List of metadata dictionaries
        #     batch_size: Number of vectors to upsert at once
        
        # Returns:
        #     int: Number of successfully added artworks

        total = len(artwork_ids)

        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)

            # Prepare batch
            vectors_to_upsert = []
            for j in range(i, batch_end):
                vectors_to_upsert.append({
                    "id": artwork_ids[j],
                    "values": embeddings[j].tolist(),
                    "metadata": metadata_list[j]
                })
        
            self.index.upsert(vectors=vectors_to_upsert)


    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_metadata: bool = True
    ):

    # Search for similar artworks.
    
    # Args:
    #     query_embedding: Query vector (numpy array)
    #     top_k: Number of results to return
    #     filter_dict: Metadata filters, e.g., {'artist': 'Monet'}
    #     include_metadata: Whether to include metadata in results
    
    # Returns:
    #     List of dictionaries with 'id', 'score', and 'metadata'

        # Convert to list
        query_vector = query_embedding.tolist()

        # Search 
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=include_metadata
        )

        # Format results
        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })

        return matches