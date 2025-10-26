"""
Script to process Kaggle dataset and upload embeddings to Pinecone.

This script:
1. Reads images from your Kaggle dataset
2. Encodes them using the CNN encoder
3. Uploads embeddings + metadata to Pinecone

Usage:
    python scripts/upload_kaggle_to_pinecone.py --dataset-path /path/to/kaggle/dataset
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import hashlib
import json
import unicodedata
import re

# Add parent directory to path to import from project
sys.path.append(str(Path(__file__).parent.parent))

from services.encoder_service import EncoderService
from services.pinecone_service import PineconeService
from config import settings


def make_ascii_safe_id(text: str) -> str:
    """
    Convert text to ASCII-safe ID for Pinecone.

    Handles special characters like accents (é, ñ, etc.) by converting to ASCII.

    Args:
        text: Original text (may contain unicode)

    Returns:
        ASCII-safe string suitable for Pinecone IDs
    """
    # Normalize unicode characters (é -> e, ñ -> n, etc.)
    text = unicodedata.normalize('NFKD', text)

    # Encode to ASCII, ignoring non-ASCII chars
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Replace spaces and special chars with underscores
    text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)

    # Remove consecutive underscores
    text = re.sub(r'_+', '_', text)

    # Remove leading/trailing underscores
    text = text.strip('_')

    return text


class KaggleDatasetProcessor:
    """Process Kaggle artist dataset and upload to Pinecone"""

    def __init__(self, dataset_path: str, batch_size: int = 100):
        """
        Args:
            dataset_path: Path to Kaggle dataset root folder
            batch_size: Number of images to encode before uploading
        """
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.encoder = EncoderService()
        self.pinecone = PineconeService()

        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.error_log = []

    def get_image_files(self):
        """
        Scan dataset and return list of (image_path, artist_name) tuples.

        Supports multiple folder structures:
        - dataset/artist_name/*.jpg
        - dataset/images/*.jpg (with metadata CSV)
        """
        image_files = []

        # Method 1: Artist folders (most common)
        # Structure: dataset/Van_Gogh/painting1.jpg
        artist_folders = [d for d in self.dataset_path.iterdir() if d.is_dir()]

        if artist_folders:
            print(f"Found {len(artist_folders)} artist folders")
            for artist_folder in artist_folders:
                artist_name = artist_folder.name.replace('_', ' ')

                # Find all image files in this artist's folder
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in artist_folder.glob(ext):
                        image_files.append((img_path, artist_name))

        # Method 2: Flat structure with CSV (if no folders found)
        # You can add CSV parsing here if needed

        return image_files

    def generate_id(self, image_path: Path, artist_name: str) -> str:
        """
        Generate unique ID for each image.
        Uses hash of filepath to ensure uniqueness.
        """
        unique_string = f"{artist_name}_{image_path.name}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def process_batch(self, batch: list):
        """
        Process a batch of images: encode and prepare for upload.

        Args:
            batch: List of (image_path, artist_name) tuples

        Returns:
            embeddings, ids, metadata lists
        """
        embeddings = []
        ids = []
        metadata = []

        for img_path, artist_name in batch:
            try:
                # Load and encode image
                image = Image.open(img_path).convert('RGB')
                embedding = self.encoder.encode_image(image)

                # Generate unique ID
                vector_id = self.generate_id(img_path, artist_name)

                # Extract individual artist name from filename
                # Format: "artist-name_painting-title.jpg" -> "Artist Name"
                individual_artist = img_path.stem.split('_')[0].replace('-', ' ').title()

                # Create metadata
                meta = {
                    'artist_name': individual_artist,  # Individual artist (e.g., "Vincent Van Gogh")
                    'art_movement': artist_name,  # Art style/movement (e.g., "Impressionism")
                    'image_filename': img_path.name,
                    'image_path': str(img_path.relative_to(self.dataset_path))
                }

                embeddings.append(embedding)
                ids.append(vector_id)
                metadata.append(meta)
                self.total_processed += 1

            except Exception as e:
                self.total_errors += 1
                error_msg = f"Error processing {img_path}: {str(e)}"
                self.error_log.append(error_msg)
                print(f"⚠️  {error_msg}")

        return np.array(embeddings), ids, metadata

    def process_and_upload(self):
        """Main processing pipeline"""
        print("=" * 60)
        print("Kaggle Dataset → Pinecone Upload Pipeline")
        print("=" * 60)

        # Step 1: Create index if needed
        print("\n[1/4] Checking Pinecone index...")
        self.pinecone.create_index(dimension=settings.embedding_dim)

        # Step 2: Scan dataset
        print("\n[2/4] Scanning dataset...")
        image_files = self.get_image_files()
        print(f"Found {len(image_files)} images to process")

        if len(image_files) == 0:
            print("❌ No images found! Check your dataset path.")
            return

        # Step 3: Process in batches
        print(f"\n[3/4] Processing and encoding images...")
        print(f"Batch size: {self.batch_size}")

        batch = []
        with tqdm(total=len(image_files), desc="Processing") as pbar:
            for img_path, artist_name in image_files:
                batch.append((img_path, artist_name))

                # Process batch when full
                if len(batch) >= self.batch_size:
                    embeddings, ids, metadata = self.process_batch(batch)

                    # Upload to Pinecone
                    if len(embeddings) > 0:
                        self.pinecone.upsert_embeddings(embeddings, ids, metadata)

                    pbar.update(len(batch))
                    batch = []

            # Process remaining batch
            if batch:
                embeddings, ids, metadata = self.process_batch(batch)
                if len(embeddings) > 0:
                    self.pinecone.upsert_embeddings(embeddings, ids, metadata)
                pbar.update(len(batch))

        # Step 4: Compute and upload artist centroids
        print("\n[4/5] Computing artist style centroids...")
        self.compute_and_upload_artist_centroids()

        # Step 5: Summary
        print("\n[5/5] Upload complete!")
        print("=" * 60)
        print(f"✅ Successfully processed: {self.total_processed}")
        print(f"❌ Errors: {self.total_errors}")

        # Show index stats
        stats = self.pinecone.get_index_stats()
        print(f"\nPinecone Index Stats:")
        print(f"  Total vectors: {stats.total_vector_count}")
        print(f"  Dimensions: {stats.dimension}")

        # Save error log if any
        if self.error_log:
            error_file = Path("upload_errors.log")
            with open(error_file, 'w') as f:
                f.write('\n'.join(self.error_log))
            print(f"\n⚠️  Error log saved to: {error_file}")

        print("=" * 60)

    def compute_and_upload_artist_centroids(self):
        """
        Compute artist centroid embeddings and upload them.

        This creates a "hero" style vector for each artist by averaging
        all their artwork embeddings.
        """
        from collections import defaultdict

        # Re-scan and process by artist
        artist_embeddings = defaultdict(list)

        print("Re-processing dataset to compute artist centroids...")
        image_files = self.get_image_files()

        # Group by artist and encode
        for img_path, artist_name in tqdm(image_files, desc="Computing centroids"):
            try:
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                embedding = self.encoder.encode_image(image)

                # Extract individual artist name
                individual_artist = img_path.stem.split('_')[0].replace('-', ' ').title()
                artist_embeddings[individual_artist].append(embedding)

            except Exception as e:
                # Skip errors, already logged during main upload
                pass

        # Compute centroids and upload
        centroid_embeddings = []
        centroid_ids = []
        centroid_metadata = []

        for artist_name, embeddings in artist_embeddings.items():
            if len(embeddings) > 0:
                # Average all embeddings for this artist
                embeddings_array = np.stack(embeddings)
                centroid = np.mean(embeddings_array, axis=0)

                # Normalize
                centroid = centroid / np.linalg.norm(centroid)

                centroid_embeddings.append(centroid)
                # Create ASCII-safe ID for Pinecone
                safe_artist_id = make_ascii_safe_id(artist_name)
                centroid_ids.append(f"artist_centroid_{safe_artist_id}")
                centroid_metadata.append({
                    'artist_name': artist_name,  # Keep original name in metadata
                    'is_centroid': True,
                    'artwork_count': len(embeddings),
                    'type': 'artist_style'
                })

                print(f"  {artist_name}: {len(embeddings)} artworks averaged")

        if centroid_embeddings:
            print(f"\nUploading {len(centroid_embeddings)} artist centroids...")
            self.pinecone.upsert_embeddings(
                embeddings=np.array(centroid_embeddings),
                ids=centroid_ids,
                metadata=centroid_metadata
            )
            print("✅ Artist centroids uploaded!")


def main():
    parser = argparse.ArgumentParser(
        description="Process Kaggle artist dataset and upload to Pinecone"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to Kaggle dataset root folder'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of images to process before uploading (default: 100)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan dataset without uploading to Pinecone'
    )

    args = parser.parse_args()

    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Process dataset
    processor = KaggleDatasetProcessor(
        dataset_path=str(dataset_path),
        batch_size=args.batch_size
    )

    if args.dry_run:
        print("DRY RUN MODE - scanning only, no upload")
        image_files = processor.get_image_files()
        print(f"\nFound {len(image_files)} images")

        # Show artist distribution
        from collections import Counter
        artists = [artist for _, artist in image_files]
        artist_counts = Counter(artists)
        print(f"\nArtist distribution (top 10):")
        for artist, count in artist_counts.most_common(10):
            print(f"  {artist}: {count} images")
    else:
        processor.process_and_upload()


if __name__ == "__main__":
    main()
