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

# Add parent directory to path to import from project
sys.path.append(str(Path(__file__).parent.parent))

from services.encoder_service import EncoderService
from services.pinecone_service import PineconeService
from config import settings


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

                # Create metadata
                meta = {
                    'artist_name': artist_name,
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

        # Step 4: Summary
        print("\n[4/4] Upload complete!")
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
