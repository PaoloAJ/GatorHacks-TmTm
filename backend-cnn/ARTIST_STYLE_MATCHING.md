# Artist Style Matching

This document explains the artist-level style matching features in your CNN backend.

## Overview

Previously, your system compared uploaded images to **individual artworks** in the database. Now you can match images to **artist styles** as a whole, identifying which artist's overall aesthetic best matches an uploaded image.

## How It Works

### Problem: Image-Level vs Artist-Level Matching

**Before (Image-Level):**
- User uploads image → finds most similar individual paintings
- Returns: "Starry Night by Van Gogh", "Sunflowers by Van Gogh", etc.
- Issue: Doesn't tell you which **artist's style** matches best

**Now (Artist-Level):**
- User uploads image → finds which artist's overall style matches
- Returns: "Van Gogh (95% match)", "Monet (87% match)", etc.
- Shows which artist's **aesthetic/style** is most similar

## Two Matching Methods

### Method 1: Aggregation (Slower but No Pre-processing)

**How it works:**
1. Queries for top N similar individual artworks (e.g., 50 images)
2. Groups results by artist
3. Aggregates similarity scores to rank artists
4. Returns artists ranked by overall style match

**Pros:**
- Works immediately with existing data
- No pre-processing needed

**Cons:**
- Slower (queries many vectors)
- Uses more Pinecone queries

**API Endpoint:** `POST /images/match-artist-style`

**Example:**
```bash
curl -X POST "http://localhost:8000/images/match-artist-style" \
  -F "file=@my_artwork.jpg" \
  -F "top_k=5" \
  -F "sample_size=50"
```

**Response:**
```json
{
  "query_filename": "my_artwork.jpg",
  "matching_artists": [
    {
      "artist_name": "Vincent Van Gogh",
      "style_match_score": 0.89,
      "confidence": 0.67,
      "artwork_count": 12,
      "max_similarity": 0.94,
      "top_artwork": "starry-night.jpg",
      "sample_artworks": [...]
    },
    {
      "artist_name": "Claude Monet",
      "style_match_score": 0.82,
      "confidence": 0.52,
      "artwork_count": 8,
      ...
    }
  ],
  "total_artists_found": 5,
  "sample_size": 50
}
```

### Method 2: Centroids (Fast but Requires Pre-processing)

**How it works:**
1. Pre-compute a "hero" style vector for each artist (average of all their works)
2. Store these artist centroids in Pinecone with special metadata
3. Query directly against artist centroids
4. Much faster - only queries K vectors instead of 50+

**Pros:**
- Very fast queries
- Efficient - one vector per artist
- Each artist has a single representative "style fingerprint"

**Cons:**
- Requires running centroid computation script first
- Need to re-compute if dataset changes

**API Endpoint:** `POST /images/match-artist-style-fast`

**Example:**
```bash
curl -X POST "http://localhost:8000/images/match-artist-style-fast" \
  -F "file=@my_artwork.jpg" \
  -F "top_k=5"
```

**Response:**
```json
{
  "query_filename": "my_artwork.jpg",
  "matching_artists": [
    {
      "artist_name": "Vincent Van Gogh",
      "style_match_score": 0.91,
      "artwork_count": 150,
      "match_type": "centroid"
    },
    {
      "artist_name": "Claude Monet",
      "style_match_score": 0.84,
      "artwork_count": 120,
      "match_type": "centroid"
    }
  ],
  "total_artists_found": 5,
  "method": "centroid"
}
```

## Setup Instructions

### Using Method 1 (Aggregation) - No Setup Needed

If you've already uploaded your dataset to Pinecone, this works immediately!

### Using Method 2 (Centroids) - Requires Pre-computation

**Option A: Upload dataset with centroid computation**

The upload script now automatically computes artist centroids:

```bash
cd backend-cnn
python scripts/upload_kaggle_to_pinecone.py \
  --dataset-path /path/to/kaggle/dataset \
  --batch-size 100
```

This will:
1. Upload all individual artworks (as before)
2. **NEW:** Compute average embedding for each artist
3. **NEW:** Upload artist centroids to Pinecone with metadata flag `is_centroid: true`

**Option B: Compute centroids from existing dataset**

If you've already uploaded artworks but want to add centroids:

```bash
python scripts/compute_artist_centroids.py \
  --dataset-path /path/to/kaggle/dataset
```

## Comparison

| Feature | Aggregation Method | Centroid Method |
|---------|-------------------|-----------------|
| **Speed** | Slower (~500ms) | Very Fast (~50ms) |
| **Setup** | None required | Run centroid script |
| **Accuracy** | Considers all artworks | Average representation |
| **Pinecone Usage** | High (queries 50+ vectors) | Low (queries K vectors) |
| **Best For** | Small datasets, exploration | Production, large datasets |

## Metadata Structure

### Individual Artwork Vectors
```json
{
  "id": "hash123456",
  "metadata": {
    "artist_name": "Vincent Van Gogh",
    "art_movement": "Post-Impressionism",
    "image_filename": "starry-night.jpg",
    "image_path": "Van_Gogh/starry-night.jpg"
  }
}
```

### Artist Centroid Vectors
```json
{
  "id": "artist_centroid_Vincent_Van_Gogh",
  "metadata": {
    "artist_name": "Vincent Van Gogh",
    "is_centroid": true,
    "artwork_count": 150,
    "type": "artist_style"
  }
}
```

## Querying with Filters

You can also filter by specific metadata:

```python
# Find similar Van Gogh paintings only
results = pinecone_service.query_similar(
    query_embedding=embedding,
    top_k=10,
    filter_dict={'artist_name': 'Vincent Van Gogh'}
)

# Query only artist centroids
results = pinecone_service.query_similar(
    query_embedding=embedding,
    top_k=5,
    filter_dict={'is_centroid': True}
)
```

## Which Method Should I Use?

**Use Aggregation if:**
- You want to get started quickly
- Your dataset is small (< 10,000 images)
- You want to see which specific artworks match

**Use Centroids if:**
- You have a large dataset
- You need fast queries for production
- You want a clean "one vector per artist" representation
- You don't mind running a pre-processing step

## Example Use Cases

### Use Case 1: "Which artist's style matches my drawing?"
→ Use **centroid method** for fast, clean results

### Use Case 2: "Show me similar artworks and group by artist"
→ Use **aggregation method** to see both artworks and artist rankings

### Use Case 3: "Is this painting more Van Gogh or Monet?"
→ Use **centroid method** with `top_k=2` to compare specific artists
