# Backend API - Style Similarity

FastAPI backend for image encoding and style similarity analysis using CNN.

## Project Structure

```
backend-cnn/
├── main.py                 # Application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── models/
│   ├── __init__.py
│   └── cnn_encoder.py    # CNN model architecture
├── routes/
│   ├── __init__.py
│   └── images.py         # Image processing endpoints
├── services/
│   ├── __init__.py
│   └── encoder_service.py # Business logic layer
└── schemas/
    ├── __init__.py
    └── image_schemas.py  # Request/Response models
```

## Setup

### With Conda (Recommended)

1. Create conda environment:
```bash
cd backend-cnn
conda create -n gatorhacks python=3.10
conda activate gatorhacks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

### With venv

1. Create virtual environment:
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

## API Endpoints

Base URL: `http://localhost:8000`

All image endpoints are prefixed with `/api/v1/images`

### `GET /`
Root health check endpoint.

**Response:**
```json
{
  "message": "Style Similarity API",
  "version": "1.0.0",
  "status": "running"
}
```

### `POST /api/v1/images/encode`
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

### `POST /api/v1/images/batch-encode`
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
    }
  ]
}
```

### `POST /api/v1/images/similarity`
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

### Layer Separation

**Routes** (`routes/images.py`)
- Handle HTTP requests and responses
- Input validation
- No direct access to models

**Services** (`services/encoder_service.py`)
- Business logic layer
- Singleton pattern for encoder instance
- Abstracts model access from routes

**Models** (`models/cnn_encoder.py`)
- CNN architecture (ResNet50 + projection head)
- Protected from direct route access
- Handles all ML operations

**Schemas** (`schemas/image_schemas.py`)
- Pydantic models for request/response validation
- Type safety

### CNN Encoder

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

## Configuration

Settings can be modified in `config.py`:

```python
# Model Settings
embedding_dim: int = 512
use_pretrained: bool = True

# Server Settings
host: str = "0.0.0.0"
port: int = 8000

# CORS Settings
cors_origins: List[str] = ["http://localhost:3000"]
```

## Development

### Interactive API Documentation
Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Adding New Routes
1. Create route file in `routes/`
2. Define router with `APIRouter()`
3. Include router in `main.py`

### Adding New Services
1. Create service file in `services/`
2. Import in routes as needed
3. Keep business logic separate from routes
