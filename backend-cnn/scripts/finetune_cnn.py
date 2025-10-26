#!/usr/bin/env python3
"""
Fine-tune the CNN encoder on your art dataset using contrastive learning.

This will improve similarity matching by training the model to recognize
artistic styles rather than just ImageNet objects.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_encoder import CNNEncoder


class ArtPairDataset(Dataset):
    """
    Dataset that generates pairs of images for contrastive learning.

    Positive pairs: Two images from the same artist
    Negative pairs: Two images from different artists
    """

    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.artist_images = {}  # artist_name -> list of image paths

        # Use same transform as encoder
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # Scan dataset and group by artist
        print("Scanning dataset...")
        for movement_folder in self.dataset_path.iterdir():
            if not movement_folder.is_dir():
                continue

            for img_path in movement_folder.glob('*.jpg'):
                # Extract artist name from filename
                artist_name = img_path.stem.split('_')[0]
                if artist_name not in self.artist_images:
                    self.artist_images[artist_name] = []
                self.artist_images[artist_name].append(img_path)

        # Filter artists with at least 2 images
        self.artist_images = {
            artist: images
            for artist, images in self.artist_images.items()
            if len(images) >= 2
        }

        self.artists = list(self.artist_images.keys())
        print(f"Found {len(self.artists)} artists with 2+ images")

    def __len__(self):
        # Arbitrary length - we generate pairs on the fly
        return len(self.artists) * 100

    def __getitem__(self, idx):
        """
        Return a pair of images and a label (1=same artist, 0=different)
        """
        # 50% positive pairs, 50% negative pairs
        is_positive = random.random() > 0.5

        if is_positive:
            # Same artist - positive pair
            artist = random.choice(self.artists)
            img1_path, img2_path = random.sample(self.artist_images[artist], 2)
            label = 1.0
        else:
            # Different artists - negative pair
            artist1, artist2 = random.sample(self.artists, 2)
            img1_path = random.choice(self.artist_images[artist1])
            img2_path = random.choice(self.artist_images[artist2])
            label = 0.0

        # Load and transform images to tensors
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # Transform to tensors
            img1_tensor = self.transform(img1)
            img2_tensor = self.transform(img2)
        except Exception:
            # If error, return random valid pair
            return self.__getitem__(random.randint(0, len(self) - 1))

        return img1_tensor, img2_tensor, torch.tensor(label, dtype=torch.float32)


def contrastive_loss(embedding1, embedding2, label, margin=0.5):
    """
    Contrastive loss for similarity learning.

    Args:
        embedding1, embedding2: L2-normalized embeddings
        label: 1 if same artist, 0 if different
        margin: Margin for negative pairs

    Returns:
        Loss value
    """
    # Cosine similarity (dot product of normalized vectors)
    similarity = torch.sum(embedding1 * embedding2, dim=1)

    # Positive pairs: maximize similarity (minimize 1 - similarity)
    positive_loss = (1 - similarity) * label

    # Negative pairs: penalize if similarity > margin
    negative_loss = torch.clamp(similarity - margin, min=0.0) * (1 - label)

    return torch.mean(positive_loss + negative_loss)


def train_epoch(encoder, dataloader, optimizer, device):
    """Train for one epoch"""
    encoder.feature_extractor.train()
    encoder.projection_head.train()

    total_loss = 0
    for img1_batch, img2_batch, labels in tqdm(dataloader, desc="Training"):
        # Move to device (already tensors from dataset)
        img1_batch = img1_batch.to(device)
        img2_batch = img2_batch.to(device)
        labels = labels.to(device)

        # Forward pass
        # Extract features
        feat1 = encoder.feature_extractor(img1_batch)
        feat2 = encoder.feature_extractor(img2_batch)

        # Project
        emb1 = encoder.projection_head(feat1)
        emb2 = encoder.projection_head(feat2)

        # Normalize
        emb1 = nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = nn.functional.normalize(emb2, p=2, dim=1)

        # Compute loss
        loss = contrastive_loss(emb1, emb2, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune CNN on art dataset")
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to art dataset'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/finetuned_encoder.pth',
        help='Output path for saved model'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Verify MPS status
    if device.type == 'mps':
        print("✅ MPS (Metal Performance Shaders) is enabled - using Apple GPU")
        print(f"   PyTorch version: {torch.__version__}")
    elif device.type == 'cuda':
        print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  WARNING: Using CPU (will be slow!)")
        print("   MPS not available - make sure you have:")
        print("   - macOS 12.3+ (Monterey or later)")
        print("   - PyTorch 1.12+ with MPS support")
        print("   - Apple Silicon Mac (M1/M2/M3)")
        response = input("\nContinue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

    # Create dataset
    dataset = ArtPairDataset(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 to avoid multiprocessing issues
    )

    # Create encoder
    print("Loading encoder...")
    encoder = CNNEncoder(embedding_dim=512, use_pretrained=True)

    # Optimizer - only train projection head, freeze feature extractor
    # (For faster training, you can unfreeze later layers if needed)
    optimizer = optim.Adam(encoder.projection_head.parameters(), lr=args.lr)

    # Check if resuming from checkpoint
    start_epoch = 0
    checkpoint_path = Path(args.output).parent / "checkpoint.pth"

    if checkpoint_path.exists():
        print(f"\n📂 Found checkpoint at {checkpoint_path}")
        response = input("Resume from checkpoint? (y/n): ").lower()
        if response == 'y':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.projection_head.load_state_dict(checkpoint['projection_head'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print(f"✅ Resuming from epoch {start_epoch + 1}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    try:
        for epoch in range(start_epoch, args.epochs):
            avg_loss = train_epoch(encoder, dataloader, optimizer, device)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'projection_head': encoder.projection_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved")

    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted! Checkpoint saved.")
        print(f"Resume later by running the same command - it will ask to resume from checkpoint.")
        return

    # Save final model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(str(output_path))
    print(f"\n✅ Model saved to: {output_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("🧹 Checkpoint cleaned up")

    print("\nTo use this model, update your encoder service to load these weights")


if __name__ == "__main__":
    main()
