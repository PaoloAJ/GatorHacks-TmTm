"""
Compute artist centroid embeddings and store in Pinecone.

This script creates a "hero" style vector for each artist by averaging
all their artwork embeddings. The centroids are stored in a separate
Pinecone namespace for fast artist-level style matching.

Usage:
    python scripts/compute_artist_centroids.py
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.pinecone_service import PineconeService
from config import settings


class ArtistCentroidComputer:
    """Compute and store artist centroid embeddings"""

    def __init__(self):
        self.pinecone = PineconeService()
        self.namespace_artworks = "artworks"  # Original artwork embeddings
        self.namespace_artists = "artist-styles"  # Artist centroid embeddings

    def fetch_all_embeddings(self):
        """
        Fetch all embeddings from Pinecone and group by artist.

        Returns:
            Dict mapping artist_name -> list of embedding vectors
        """
        print("Fetching embeddings from Pinecone...")

        # Get index stats
        stats = self.pinecone.get_index_stats()
        total_vectors = stats.total_vector_count
        dimension = stats.dimension

        print(f"Total vectors in index: {total_vectors}")
        print(f"Embedding dimension: {dimension}")

        # Pinecone fetch limitations: we need to use scroll/pagination
        # For now, we'll use a large query approach
        artist_embeddings = defaultdict(list)
        artist_metadata = {}  # Store additional info about each artist

        # Create a random query vector to fetch results
        # We'll do multiple queries with different random vectors to get diverse samples
        num_queries = max(1, total_vectors // 10000 + 1)

        for i in range(num_queries):
            # Random vector query
            random_vector = np.random.randn(dimension).astype(np.float32)
            random_vector = random_vector / np.linalg.norm(random_vector)

            results = self.pinecone.query_similar(
                query_embedding=random_vector, top_k=min(10000, total_vectors)
            )

            print(f"Query {i+1}/{num_queries}: Got {len(results)} results")

            # We don't get vectors back from query, so we need a different approach
            # Let's use the Pinecone fetch API instead

        # Alternative: Use fetch API
        # Note: This requires knowing the IDs, which we don't have
        # Better approach: During upload, also upload to artist namespace

        print("\n⚠️  Note: To efficiently compute artist centroids, we need to:")
        print("1. Store vectors in a namespace during upload, OR")
        print("2. Re-process the original dataset and compute centroids directly")
        print(
            "\nRecommended: Run the improved upload script that computes centroids on-the-fly"
        )

        return artist_embeddings

    def compute_centroids_from_dataset(self, dataset_path: str):
        """
        Compute artist centroids directly from the original dataset.

        This is more efficient than fetching from Pinecone.

        Args:
            dataset_path: Path to the Kaggle dataset
        """
        from PIL import Image
        from services.encoder_service import EncoderService

        encoder = EncoderService()
        dataset_path = Path(dataset_path)

        # Dictionary to accumulate embeddings by artist
        artist_embeddings = defaultdict(list)

        print("Processing dataset to compute artist centroids...")

        # Find all artist folders
        artist_folders = [d for d in dataset_path.iterdir() if d.is_dir()]

        for artist_folder in tqdm(artist_folders, desc="Artists"):
            artist_name = artist_folder.name.replace("_", " ")

            # Process all images for this artist
            image_count = 0
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                for img_path in artist_folder.glob(ext):
                    try:
                        image = Image.open(img_path).convert("RGB")
                        embedding = encoder.encode_image(image)
                        artist_embeddings[artist_name].append(embedding)
                        image_count += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

            print(f"  {artist_name}: {image_count} images")

        # Compute centroids (average embeddings)
        print("\nComputing centroids...")
        artist_centroids = {}

        for artist_name, embeddings in artist_embeddings.items():
            if len(embeddings) > 0:
                # Stack and average
                embeddings_array = np.stack(embeddings)
                centroid = np.mean(embeddings_array, axis=0)

                # Normalize the centroid
                centroid = centroid / np.linalg.norm(centroid)

                artist_centroids[artist_name] = centroid

                print(f"  {artist_name}: {len(embeddings)} artworks -> 1 centroid")

        return artist_centroids

    def upload_centroids_to_pinecone(self, artist_centroids: dict):
        """
        Upload artist centroid embeddings to Pinecone in a separate namespace.

        Args:
            artist_centroids: Dict mapping artist_name -> centroid embedding
        """
        print("\nUploading artist centroids to Pinecone...")

        embeddings = []
        ids = []
        metadata = []

        for artist_name, centroid in artist_centroids.items():
            embeddings.append(centroid)
            ids.append(f"artist_{artist_name.replace(' ', '_')}")
            metadata.append(
                {
                    "artist_name": artist_name,
                    "is_centroid": True,
                    "type": "artist_style",
                }
            )

        embeddings_array = np.stack(embeddings)

        # Upload to Pinecone
        # Note: You may need to modify PineconeService to support namespaces
        self.pinecone.upsert_embeddings(
            embeddings=embeddings_array, ids=ids, metadata=metadata
        )

        print(f"✅ Uploaded {len(artist_centroids)} artist centroids")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute artist centroid embeddings")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to original Kaggle dataset (recommended for accuracy)",
    )
    parser.add_argument(
        "--from-pinecone",
        action="store_true",
        help="Compute from existing Pinecone vectors (not recommended)",
    )

    args = parser.parse_args()

    computer = ArtistCentroidComputer()

    if args.dataset_path:
        # Compute from original dataset (recommended)
        centroids = computer.compute_centroids_from_dataset(args.dataset_path)
        computer.upload_centroids_to_pinecone(centroids)

    elif args.from_pinecone:
        # Fetch from Pinecone (less efficient)
        artist_embeddings = computer.fetch_all_embeddings()
        # Compute centroids...
        print("⚠️  Not fully implemented - use --dataset-path instead")

    else:
        print("Error: Please specify --dataset-path or --from-pinecone")
        sys.exit(1)


if __name__ == "__main__":
    main()
