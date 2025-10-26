#!/usr/bin/env python3
"""
Diagnostic script to understand similarity scores and confidence issues.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.encoder_service import EncoderService
from services.pinecone_service import PineconeService
from services.artist_style_service import ArtistStyleService


def diagnose_similarity_scores(test_image_path: str):
    """
    Run comprehensive diagnostics on similarity scoring.

    Args:
        test_image_path: Path to a test image
    """
    print("=" * 70)
    print("SIMILARITY SCORE DIAGNOSTICS")
    print("=" * 70)

    # Initialize services
    encoder = EncoderService()
    pinecone = PineconeService()
    artist_style = ArtistStyleService()

    # Load and encode test image
    print(f"\n1. Loading test image: {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    embedding = encoder.encode_image(image)

    # Check embedding properties
    print(f"\n2. Embedding Properties:")
    print(f"   Shape: {embedding.shape}")
    print(f"   L2 Norm: {np.linalg.norm(embedding):.6f}")
    print(f"   Min value: {embedding.min():.6f}")
    print(f"   Max value: {embedding.max():.6f}")
    print(f"   Mean value: {embedding.mean():.6f}")

    if abs(np.linalg.norm(embedding) - 1.0) > 0.01:
        print("   ⚠️  WARNING: Embedding is not normalized! (should be ~1.0)")
    else:
        print("   ✅ Embedding is properly normalized")

    # Test self-similarity (should be ~1.0)
    print(f"\n3. Self-Similarity Test:")
    self_similarity = np.dot(embedding, embedding)
    print(f"   Embedding · Embedding = {self_similarity:.6f}")

    if abs(self_similarity - 1.0) > 0.01:
        print("   ⚠️  WARNING: Self-similarity should be ~1.0 for normalized vectors")
    else:
        print("   ✅ Self-similarity is correct")

    # Query Pinecone for similar artworks
    print(f"\n4. Querying Pinecone for similar artworks...")
    artwork_results = pinecone.query_similar(
        query_embedding=embedding,
        top_k=10,
        filter_dict={'is_centroid': {'$ne': True}}  # Exclude centroids
    )

    print(f"   Found {len(artwork_results)} results")
    if artwork_results:
        print(f"   Top 5 similar artworks:")
        for i, result in enumerate(artwork_results[:5], 1):
            score = result['score']
            artist = result['metadata'].get('artist_name', 'Unknown')
            filename = result['metadata'].get('image_filename', 'Unknown')
            print(f"   {i}. {artist:30s} - {filename:40s} (score: {score:.4f})")

        # Analyze score distribution
        scores = [r['score'] for r in artwork_results]
        print(f"\n   Score Statistics:")
        print(f"   - Highest: {max(scores):.4f}")
        print(f"   - Lowest:  {min(scores):.4f}")
        print(f"   - Average: {np.mean(scores):.4f}")
        print(f"   - Std Dev: {np.std(scores):.4f}")

        if max(scores) < 0.5:
            print("\n   ⚠️  LOW SIMILARITY SCORES DETECTED!")
            print("   Possible causes:")
            print("   - Image is very different from training data")
            print("   - Model not trained/fine-tuned on this dataset")
            print("   - Using pre-trained ResNet (ImageNet) without fine-tuning")
    else:
        print("   ❌ No results found!")

    # Query for artist centroids
    print(f"\n5. Querying Pinecone for artist centroids...")
    centroid_results = pinecone.query_similar(
        query_embedding=embedding,
        top_k=10,
        filter_dict={'is_centroid': True}
    )

    print(f"   Found {len(centroid_results)} centroid results")
    if centroid_results:
        print(f"   Top 5 matching artists (centroid method):")
        for i, result in enumerate(centroid_results[:5], 1):
            score = result['score']
            artist = result['metadata'].get('artist_name', 'Unknown')
            count = result['metadata'].get('artwork_count', 0)
            print(f"   {i}. {artist:30s} ({count:3d} artworks) - score: {score:.4f}")

        # Analyze centroid scores
        scores = [r['score'] for r in centroid_results]
        print(f"\n   Centroid Score Statistics:")
        print(f"   - Highest: {max(scores):.4f}")
        print(f"   - Lowest:  {min(scores):.4f}")
        print(f"   - Average: {np.mean(scores):.4f}")
    else:
        print("   ❌ No centroid results found!")
        print("   💡 Run the upload script with centroid computation first")

    # Test aggregation method
    print(f"\n6. Testing Aggregation Method...")
    try:
        agg_results = artist_style.find_artist_by_aggregation(
            query_embedding=embedding,
            top_k=5,
            sample_size=50
        )

        print(f"   Top 5 artists (aggregation method):")
        for i, result in enumerate(agg_results, 1):
            artist = result['artist_name']
            style_score = result['style_match_score']
            confidence = result['confidence']
            count = result['artwork_count']
            print(f"   {i}. {artist:30s}")
            print(f"      Style Score: {style_score:.4f} | Confidence: {confidence:.4f} | Count: {count}")

        if agg_results and agg_results[0]['confidence'] < 0.3:
            print("\n   ⚠️  LOW CONFIDENCE DETECTED!")
            print("   This is normal if:")
            print("   - Your test image is very different from the dataset")
            print("   - Using pre-trained ResNet without fine-tuning")
            print("   - Dataset has many diverse styles")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    if artwork_results and max([r['score'] for r in artwork_results]) < 0.6:
        print("\n🔧 LOW SIMILARITY SCORES - Possible Solutions:")
        print("   1. Fine-tune the CNN on your specific dataset")
        print("   2. Use a pre-trained model trained on artwork (not ImageNet)")
        print("   3. Adjust your expectations - art similarity is subjective")
        print("   4. Collect more diverse training data")

    if centroid_results and max([r['score'] for r in centroid_results]) > 0.7:
        print("\n✅ CENTROID SCORES LOOK GOOD")
        print("   Consider using the fast centroid method for production")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose similarity scoring issues")
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to test image'
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    diagnose_similarity_scores(str(image_path))


if __name__ == "__main__":
    main()
