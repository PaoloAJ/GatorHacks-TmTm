# Backend API - Style Similarity

FastAPI backend for image encoding and style similarity analysis using CNN.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Style Similarity API",
  "status": "running"
}
```

### `POST /encode`
Encode a single image into a feature vector.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "embedding": [0.123, 0.456, ...],
  "shape": [512]
}
```

### `POST /batch-encode`
Encode multiple images in a batch.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `files` (multiple image files)

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "embedding": [0.123, ...],
      "shape": [512]
    },
    ...
  ]
}
```

### `POST /similarity`
Compute cosine similarity between two images.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file1`, `file2` (two image files)

**Response:**
```json
{
  "similarity": 0.87,
  "file1": "image1.jpg",
  "file2": "image2.jpg"
}
```

## Architecture

### CNN Encoder (`models/cnn_encoder.py`)

The encoder uses a pre-trained ResNet50 backbone with a custom projection head:

1. **Feature Extraction**: ResNet50 (pre-trained on ImageNet) extracts 2048-dimensional features
2. **Projection Head**: Projects features to 512-dimensional embeddings
3. **Normalization**: L2 normalization for cosine similarity comparison

**Key Features:**
- GPU support (automatically uses CUDA if available)
- Batch processing support
- Model weight saving/loading
- Custom CNN architecture option (`CustomCNN` class)

### Image Preprocessing
- Resize to 256x256
- Center crop to 224x224
- Normalize with ImageNet statistics

## Development

The API includes CORS middleware configured for Next.js frontend (port 3000).

To modify the embedding dimension, update the `CNNEncoder` initialization in `main.py`:
```python
encoder = CNNEncoder(embedding_dim=512)  # Change 512 to desired dimension
```
