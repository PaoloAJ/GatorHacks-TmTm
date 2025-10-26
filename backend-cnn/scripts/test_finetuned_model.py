#!/usr/bin/env python3
"""
Quick test script to compare before/after fine-tuning results.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_encoder import CNNEncoder
from services.pinecone_service import PineconeService


def test_model(image_path: str, model_weights_path: str = None):
    """
    Test a model (fine-tuned or not) on an image.

    Args:
        image_path: Path to test image
        model_weights_path: Path to fine-tuned weights (None = use pretrained)
    """
    print("=" * 70)
    if model_weights_path:
        print(f"TESTING FINE-TUNED MODEL: {model_weights_path}")
    else:
        print("TESTING PRE-TRAINED MODEL (baseline)")
    print("=" * 70)

    # Load encoder
    encoder = CNNEncoder(embedding_dim=512, use_pretrained=True)
    if model_weights_path:
        encoder.load_weights(model_weights_path)
        print("✅ Loaded fine-tuned weights")
    else:
        print("✅ Using pre-trained ImageNet weights")

    # Load image
    print(f"\nTest image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    embedding = encoder.encode_image(image)

    # Query Pinecone for similar artworks
    print("\n📊 Querying Pinecone for similar artworks...")
    pinecone = PineconeService()

    artwork_results = pinecone.query_similar(
        query_embedding=embedding,
        top_k=10,
        filter_dict={'is_centroid': {'$ne': True}}
    )

    print("\nTop 5 Similar Artworks:")
    print("-" * 70)
    for i, result in enumerate(artwork_results[:5], 1):
        score = result['score']
        artist = result['metadata'].get('artist_name', 'Unknown')
        filename = result['metadata'].get('image_filename', 'Unknown')
        print(f"{i}. {artist:30s} - {filename:30s} [{score:.4f}]")

    # Query for artist centroids
    print("\n📊 Querying for artist style matches...")
    centroid_results = pinecone.query_similar(
        query_embedding=embedding,
        top_k=5,
        filter_dict={'is_centroid': True}
    )

    print("\nTop 5 Matching Artists:")
    print("-" * 70)
    for i, result in enumerate(centroid_results[:5], 1):
        score = result['score']
        artist = result['metadata'].get('artist_name', 'Unknown')
        count = result['metadata'].get('artwork_count', 0)
        print(f"{i}. {artist:30s} ({count:3d} works) [{score:.4f}]")

    # Calculate statistics
    artwork_scores = [r['score'] for r in artwork_results]
    centroid_scores = [r['score'] for r in centroid_results]

    print("\n📈 Score Statistics:")
    print("-" * 70)
    print(f"Artwork Matches:")
    print(f"  - Max:  {max(artwork_scores):.4f}")
    print(f"  - Avg:  {np.mean(artwork_scores):.4f}")
    print(f"  - Min:  {min(artwork_scores):.4f}")

    print(f"\nArtist Centroids:")
    print(f"  - Max:  {max(centroid_scores):.4f}")
    print(f"  - Avg:  {np.mean(centroid_scores):.4f}")
    print(f"  - Min:  {min(centroid_scores):.4f}")

    print("=" * 70)
    print()

    return {
        'max_artwork_score': max(artwork_scores),
        'avg_artwork_score': np.mean(artwork_scores),
        'max_centroid_score': max(centroid_scores),
        'avg_centroid_score': np.mean(centroid_scores)
    }


def compare_models(image_path: str, finetuned_weights: str):
    """
    Compare pre-trained vs fine-tuned model side-by-side.
    """
    print("\n" + "=" * 70)
    print("COMPARING PRE-TRAINED vs FINE-TUNED MODEL")
    print("=" * 70 + "\n")

    # Test baseline
    baseline_results = test_model(image_path, model_weights_path=None)

    # Test fine-tuned
    finetuned_results = test_model(image_path, model_weights_path=finetuned_weights)

    # Show improvement
    print("\n" + "=" * 70)
    print("📊 IMPROVEMENT SUMMARY")
    print("=" * 70)

    artwork_improvement = (
        (finetuned_results['max_artwork_score'] - baseline_results['max_artwork_score'])
        / baseline_results['max_artwork_score'] * 100
    )
    centroid_improvement = (
        (finetuned_results['max_centroid_score'] - baseline_results['max_centroid_score'])
        / baseline_results['max_centroid_score'] * 100
    )

    print(f"\nMax Artwork Score:")
    print(f"  Baseline:    {baseline_results['max_artwork_score']:.4f}")
    print(f"  Fine-tuned:  {finetuned_results['max_artwork_score']:.4f}")
    print(f"  Improvement: {artwork_improvement:+.1f}%")

    print(f"\nMax Centroid Score:")
    print(f"  Baseline:    {baseline_results['max_centroid_score']:.4f}")
    print(f"  Fine-tuned:  {finetuned_results['max_centroid_score']:.4f}")
    print(f"  Improvement: {centroid_improvement:+.1f}%")

    if artwork_improvement > 20 or centroid_improvement > 20:
        print("\n✅ SIGNIFICANT IMPROVEMENT! Fine-tuning worked well!")
    elif artwork_improvement > 5 or centroid_improvement > 5:
        print("\n👍 MODERATE IMPROVEMENT. Consider training more epochs.")
    else:
        print("\n⚠️  LIMITED IMPROVEMENT. May need more training or data.")

    print("=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to test image'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to fine-tuned weights (omit to test baseline)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare baseline vs fine-tuned side-by-side'
    )

    args = parser.parse_args()

    if args.compare and args.model:
        compare_models(args.image, args.model)
    else:
        test_model(args.image, args.model)


if __name__ == "__main__":
    main()
