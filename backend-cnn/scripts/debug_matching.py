#!/usr/bin/env python3
"""
Debug why artist matching isn't working correctly.
"""

import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from services.encoder_service import EncoderService
from services.pinecone_service import PineconeService


def debug_matching(image_path: str):
    """
    Debug artist matching by showing detailed results.
    """
    print("=" * 80)
    print("DEBUGGING ARTIST MATCHING")
    print("=" * 80)

    # Parse expected artist from filename
    image_path = Path(image_path)
    filename = image_path.name
    expected_artist = filename.split('_')[0].replace('-', ' ').title()

    print(f"\nTest Image: {filename}")
    print(f"Expected Artist: {expected_artist}")
    print(f"Art Movement Folder: {image_path.parent.name}")

    # Encode image
    print("\n1. Encoding image...")
    encoder = EncoderService()
    image = Image.open(image_path).convert('RGB')
    embedding = encoder.encode_image(image)
    print(f"   ✅ Embedding shape: {embedding.shape}")
    print(f"   ✅ Embedding norm: {embedding.dot(embedding):.6f}")

    # Query Pinecone
    print("\n2. Querying Pinecone for similar artworks...")
    pinecone = PineconeService()

    results = pinecone.query_similar(
        query_embedding=embedding,
        top_k=20,
        filter_dict={'is_centroid': {'$ne': True}}
    )

    print(f"   Found {len(results)} results")

    # Check if expected artist is in results
    print("\n3. Top 20 Similar Artworks:")
    print("-" * 80)
    expected_artist_found = False
    expected_artist_rank = None

    for i, result in enumerate(results, 1):
        score = result['score']
        artist = result['metadata'].get('artist_name', 'Unknown')
        filename_result = result['metadata'].get('image_filename', 'Unknown')
        movement = result['metadata'].get('art_movement', 'Unknown')

        marker = ""
        if artist.lower() == expected_artist.lower():
            marker = " ⭐ EXPECTED ARTIST"
            expected_artist_found = True
            if expected_artist_rank is None:
                expected_artist_rank = i

        print(f"{i:2d}. [{score:.4f}] {artist:30s} | {movement:25s}{marker}")
        if i <= 5:
            print(f"     → {filename_result}")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    if expected_artist_rank == 1:
        print(f"✅ CORRECT! Expected artist '{expected_artist}' is rank #1")
    elif expected_artist_found:
        print(f"⚠️  Expected artist '{expected_artist}' found at rank #{expected_artist_rank}")
        print(f"   (Should be #1, but at least it's in top results)")
    else:
        print(f"❌ PROBLEM! Expected artist '{expected_artist}' NOT in top 20 results!")
        print(f"\n   Possible issues:")
        print(f"   1. Artist name mismatch during upload vs query")
        print(f"   2. Not enough artworks by this artist in dataset")
        print(f"   3. Embeddings not similar (model issue)")

    # Check artist name variations in Pinecone
    print(f"\n4. Checking for artist name variations in Pinecone...")
    print("   Artist names found in top 20 results:")
    unique_artists = set()
    for result in results:
        artist = result['metadata'].get('artist_name', 'Unknown')
        unique_artists.add(artist)

    for artist in sorted(unique_artists)[:10]:
        marker = "⭐" if artist.lower() == expected_artist.lower() else "  "
        print(f"   {marker} {artist}")

    # Query for artist centroids
    print(f"\n5. Checking artist centroids...")
    centroid_results = pinecone.query_similar(
        query_embedding=embedding,
        top_k=10,
        filter_dict={'is_centroid': True}
    )

    if centroid_results:
        print("   Top 5 matching artist styles (centroids):")
        for i, result in enumerate(centroid_results[:5], 1):
            score = result['score']
            artist = result['metadata'].get('artist_name', 'Unknown')
            count = result['metadata'].get('artwork_count', 0)
            marker = "⭐" if artist.lower() == expected_artist.lower() else ""
            print(f"   {i}. [{score:.4f}] {artist:30s} ({count} artworks) {marker}")
    else:
        print("   ❌ No artist centroids found! Did you run the centroid upload?")

    # Score statistics
    scores = [r['score'] for r in results]
    print(f"\n6. Score Statistics:")
    print(f"   Max:  {max(scores):.4f}")
    print(f"   Min:  {min(scores):.4f}")
    print(f"   Avg:  {sum(scores)/len(scores):.4f}")

    if max(scores) < 0.5:
        print("\n   ⚠️  All scores are quite low (< 0.5)")
        print("   This is expected with pre-trained ImageNet ResNet.")
        print("   Fine-tuning will improve this significantly.")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug artist matching")
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to test image from dataset'
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    debug_matching(args.image)


if __name__ == "__main__":
    main()
