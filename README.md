# ART-ificial

Discover the Artist Behind AI Art

## Table of Contents

- [Overview](#overview)
- [Team Members](#team-members)
- [Motivation](#motivation)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [License](#license)

## Overview

ART-ificial is an AI-powered web application that identifies which real artists' styles are most similar to uploaded images. By leveraging advanced computer vision techniques, it provides transparency and credit to the relationship between human artists and AI-generated artwork.

The platform uses two complementary approaches:

- **CNN-based matching**: ResNet50 with custom projection head for artist style identification
- **CLIP-based search**: Vision-language model for hybrid visual and semantic similarity

## Team Members

- [Anilov Villanueva]
- [Jason Li]
- [Skylar Liu]
- [Emlynn Rossiya]
- [Owen Brooks]

## Motivation

Art and technology should not be in opposition but should coexist harmoniously. This project bridges the gap between human creativity and AI-generated art by:

- **Providing transparency** about artistic influences in AI-generated images
- **Giving credit** to the real artists whose styles inspire AI creations
- **Respecting the relationship** between human artists and technology
- **Amplifying creativity** rather than replacing it

## Features

### Core Functionality

- **Image Upload**: Drag-and-drop or file selection interface for seamless image uploads
- **Dual Search Methods**: Choose between CNN or CLIP algorithms for different matching approaches
- **Artist Identification**: Returns top 5 matching artists with similarity scores
- **Artwork Details**: View matching artworks with metadata including title, year, and artistic style
- **Interactive Documentation**: In-app documentation explaining the technology and methodology

### Technical Features

- Fast artist style matching using pre-computed centroids (50ms response time)
- Batch image processing capabilities
- Hybrid search combining visual and semantic similarity
- CDN-hosted artwork images for fast loading
- Vector similarity search across 90,000+ artworks

## Tech Stack

### Frontend

- **Next.js 16** - React framework with App Router
- **React 19** - UI library with React Compiler for automatic optimization
- **Tailwind CSS 4** - Utility-first CSS framework
- **Shadcn/ui** - Accessible component library
- **Lucide React** - Icon system
- **Vercel** - Deployment platform

### Backend (CNN)

- **FastAPI** - Modern Python web framework
- **PyTorch 2.1** - Deep learning framework
- **ResNet50** - Pre-trained CNN with custom projection head
- **Pinecone** - Serverless vector database
- **Google Cloud Run** - Serverless container deployment

### Backend (CLIP)

- **FastAPI** - Web framework
- **OpenAI CLIP (ViT-L/14)** - Vision-language model
- **PostgreSQL + pgvector** - Vector database with hybrid search
- **Railway** - Deployment platform

### Infrastructure

- **Docker** - Containerization
- **Cloudflare Images** - CDN for artwork hosting
- **Pinecone Vector DB** - AWS us-east-1 serverless instance

## Architecture

### CNN Method (Artist Style Matching)

```
User Upload Image
    ↓
ResNet50 Feature Extraction (2048-dim)
    ↓
Projection Head (512-dim embedding)
    ↓
Query Pinecone Vector DB
    ↓
Top-K Similar Artworks Retrieved
    ↓
Aggregate by Artist or Compare to Centroids
    ↓
Return Top Artists with Scores
```

### CLIP Method (Hybrid Search)

```
User Upload Image
    ↓
CLIP Vision Transformer Encoding
    ↓
Hybrid Score = α * (visual_sim) + (1-α) * (text_sim)
    ↓
Query PostgreSQL pgvector
    ↓
Return Top Artworks with CDN URLs
```

### Key Technical Decisions

1. **Lazy Model Loading**: Singleton pattern with delayed initialization to handle Cloud Run startup timeouts
2. **Dual Matching Strategies**:
   - Aggregation method (no pre-processing, more accurate)
   - Centroid method (pre-computed, 10x faster)
3. **L2 Normalization**: All embeddings normalized for cosine similarity
4. **Confidence Scoring**: Combines average similarity, frequency, and max similarity

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- Docker (for backend deployment)
- Pinecone account
- Cloudflare Images account (or alternative CDN)

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:3000`

### Backend (CNN) Setup

```bash
cd backend-cnn
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env.yaml
# Edit .env.yaml with your Pinecone credentials

# Run the server
uvicorn main:app --reload --port 8080
```

### Backend (CLIP) Setup

```bash
cd backend-clip
pip install -r requirements.txt

# Set DATABASE_URL environment variable
export DATABASE_URL="your_postgresql_connection_string"

# Run the server
uvicorn main:app --reload --port 8000
```

## API Documentation

### CNN Backend

Base URL: `https://backend-cnn-298442359505.us-east1.run.app/api/v1`

#### Encode Single Image

```http
POST /images/encode
Content-Type: multipart/form-data

image: <file>
```

#### Match Artist Style (Fast)

```http
POST /images/match-artist-style-fast
Content-Type: multipart/form-data

image: <file>
top_k: 5
```

Response:

```json
{
  "matches": [
    {
      "artist_name": "Vincent Van Gogh",
      "similarity_score": 0.92,
      "artwork_count": 150,
      "sample_artwork": {
        "title": "Starry Night",
        "year": "1889",
        "style": "Post-Impressionism",
        "image_url": "https://cdn.example.com/..."
      }
    }
  ]
}
```

#### Match Artist Style (Accurate)

```http
POST /images/match-artist-style
Content-Type: multipart/form-data

image: <file>
top_k: 5
```

### CLIP Backend

Base URL: `https://imagesimilarity.up.railway.app`

#### Hybrid Search

```http
POST /search?alpha=0.7&top_k=5
Content-Type: multipart/form-data

file: <image>
```

Parameters:

- `alpha`: Weight for visual similarity (0-1), default 0.7
- `top_k`: Number of results to return, default 5

## Deployment

### Frontend (Vercel)

```bash
cd frontend
vercel deploy --prod
```

### Backend (Google Cloud Run)

```bash
cd backend-cnn
gcloud run deploy backend-cnn \
  --source . \
  --platform managed \
  --region us-east1 \
  --allow-unauthenticated \
  --timeout 300 \
  --memory 2Gi
```

### Backend (Railway)

```bash
cd backend-clip
# Connect Railway CLI and deploy
railway up
```

## Project Structure

```
gatorhacks/
├── frontend/                          # Next.js application
│   ├── src/app/
│   │   ├── page.js                   # Main app component
│   │   ├── layout.js                 # Root layout
│   │   └── components/
│   │       ├── ImageProcessor.js     # Results display
│   │       ├── Documentation.js      # In-app docs
│   │       └── ui/                   # Shadcn components
│   ├── package.json
│   ├── next.config.mjs
│   └── tailwind.config.js
│
├── backend-cnn/                       # FastAPI CNN service
│   ├── main.py                       # App initialization
│   ├── config.py                     # Configuration
│   ├── Dockerfile                    # Container definition
│   ├── routes/
│   │   └── images.py                 # Image endpoints
│   ├── services/
│   │   ├── encoder_service.py        # CNN wrapper
│   │   ├── pinecone_service.py       # Vector DB ops
│   │   └── artist_style_service.py   # Matching logic
│   ├── models/
│   │   └── cnn_encoder.py            # ResNet50 model
│   ├── schemas/
│   │   └── image_schemas.py          # Request/response models
│   └── scripts/                      # Data processing
│
├── backend-clip/                      # FastAPI CLIP service
│   ├── main.py                       # Search endpoint
│   └── scripts/                      # Data processing
│
└── README.md
```

## Dataset

- **Source**: WikiArt Kaggle dataset
- **Size**: Approximately 90,000 artworks
- **Coverage**: Multiple artistic movements, styles, and historical periods
- **Artists**: Hundreds of renowned artists from various eras
- **Storage**: Cloudflare Images CDN for efficient global delivery

The dataset includes metadata for each artwork:

- Artist name
- Art movement/style
- Creation year
- Artwork title
- High-resolution image

## License

This project was created for GatorHacks hackathon.
