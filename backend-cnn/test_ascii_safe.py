#!/usr/bin/env python3
"""Quick test to verify ASCII-safe ID generation"""

import unicodedata
import re


def make_ascii_safe_id(text: str) -> str:
    """Convert text to ASCII-safe ID for Pinecone."""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    return text


# Test cases
test_names = [
    "Eugène Grasset",
    "José Clemente Orozco",
    "François Boucher",
    "René Magritte",
    "Käthe Kollwitz",
    "Joan Miró",
    "Normal Artist Name"
]

print("Testing ASCII-safe ID generation:")
print("=" * 60)

for name in test_names:
    safe_id = make_ascii_safe_id(name)
    full_id = f"artist_centroid_{safe_id}"
    print(f"{name:30s} → {full_id}")

print("=" * 60)
print("✅ All IDs are ASCII-safe!")
