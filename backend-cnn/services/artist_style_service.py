"""
Artist Style Service - Aggregates artist embeddings to find style matches
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from services.pinecone_service import PineconeService


class ArtistStyleService:
    """
    Service for artist-level style matching.

    This service provides different strategies for matching uploaded images
    to artist styles rather than individual artworks.
    """

    def __init__(self):
        self.pinecone_service = PineconeService()
        self._artist_centroids = None

    def find_artist_by_aggregation(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        sample_size: int = 50
    ) -> List[Dict]:
        """
        Find artists by aggregating results from similar images.

        Strategy: Query for top N similar images, then count which artists
        appear most frequently in the results. This gives you artists whose
        overall style matches, weighted by similarity scores.

        Args:
            query_embedding: The encoded image embedding
            top_k: Number of top artists to return
            sample_size: How many similar images to sample (higher = more accurate)

        Returns:
            List of dicts with artist info and aggregated scores
        """
        # Query for many similar images
        results = self.pinecone_service.query_similar(
            query_embedding=query_embedding,
            top_k=sample_size
        )

        # Aggregate by artist
        artist_scores = defaultdict(lambda: {
            'total_score': 0.0,
            'count': 0,
            'max_score': 0.0,
            'artworks': []
        })

        for result in results:
            metadata = result['metadata']
            artist_name = metadata.get('artist_name', 'Unknown')
            score = result['score']

            artist_scores[artist_name]['total_score'] += score
            artist_scores[artist_name]['count'] += 1
            artist_scores[artist_name]['max_score'] = max(
                artist_scores[artist_name]['max_score'],
                score
            )
            artist_scores[artist_name]['artworks'].append({
                'filename': metadata.get('image_filename', ''),
                'score': score
            })

        # Compute aggregated scores for each artist
        artist_results = []
        for artist_name, data in artist_scores.items():
            # Average score weighted by frequency
            avg_score = data['total_score'] / data['count']

            # Improved confidence score calculation
            # Combines: average similarity, frequency, and max score
            frequency_weight = data['count'] / sample_size

            # Boost confidence based on both average and max similarity
            similarity_component = 0.7 * avg_score + 0.3 * data['max_score']

            # Final confidence (scaled to be more interpretable)
            # Square root of frequency gives more reasonable scaling
            confidence = similarity_component * (frequency_weight ** 0.5)

            artist_results.append({
                'artist_name': artist_name,
                'style_match_score': float(avg_score),
                'confidence': float(confidence),
                'artwork_count': data['count'],
                'max_similarity': float(data['max_score']),
                'top_artwork': data['artworks'][0]['filename'],  # Most similar piece
                'sample_artworks': data['artworks'][:3]  # Top 3 pieces
            })

        # Sort by confidence score
        artist_results.sort(key=lambda x: x['confidence'], reverse=True)

        return artist_results[:top_k]

    def compute_artist_centroids(self) -> Dict[str, np.ndarray]:
        """
        Compute centroid (average) embeddings for each artist.

        This creates a "hero" style vector for each artist by averaging
        all their artwork embeddings. This is slower but creates a persistent
        representation of each artist's style.

        Returns:
            Dict mapping artist_name -> average embedding vector
        """
        print("Computing artist centroids from Pinecone index...")

        # Get all vectors from Pinecone
        # Note: For large datasets, you may need to implement pagination
        index_stats = self.pinecone_service.get_index_stats()
        total_vectors = index_stats.total_vector_count

        print(f"Processing {total_vectors} vectors...")

        # Dictionary to collect embeddings by artist
        artist_embeddings = defaultdict(list)

        # Fetch vectors in batches using a dummy query
        # (This is a workaround - Pinecone doesn't have a direct "fetch all" API)
        # We'll create a zero vector and query with very high top_k
        dimension = index_stats.dimension
        dummy_vector = np.zeros(dimension)

        batch_size = 10000
        results = self.pinecone_service.query_similar(
            query_embedding=dummy_vector,
            top_k=min(batch_size, total_vectors)
        )

        # Group embeddings by artist
        for result in results:
            metadata = result.get('metadata', {})
            artist_name = metadata.get('artist_name', 'Unknown')

            # Note: Pinecone doesn't return vectors in query results by default
            # We'll need to fetch them separately or use a different approach
            # For now, we'll use the aggregation method above

        print("⚠️  Note: Computing centroids requires fetching all vectors from Pinecone")
        print("⚠️  This is expensive. Consider using the aggregation method instead.")
        print("⚠️  Or pre-compute centroids and store in a separate namespace.")

        return {}

    def find_artist_by_centroid(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find artists by comparing to pre-computed artist centroids.

        This queries Pinecone for artist centroid vectors (identified by
        metadata filter is_centroid=True). Much faster than aggregation
        if centroids are pre-computed.

        Args:
            query_embedding: The encoded image embedding
            top_k: Number of top artists to return

        Returns:
            List of dicts with artist info and similarity scores
        """
        # Query Pinecone for artist centroids only
        results = self.pinecone_service.query_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict={'is_centroid': True}
        )

        # Format results
        artist_matches = []
        for result in results:
            metadata = result['metadata']
            artist_matches.append({
                'artist_name': metadata.get('artist_name', 'Unknown'),
                'style_match_score': float(result['score']),
                'artwork_count': metadata.get('artwork_count', 0),
                'match_type': 'centroid'
            })

        return artist_matches


# Global service instance
artist_style_service = ArtistStyleService()
