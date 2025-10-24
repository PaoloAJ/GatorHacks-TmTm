from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List
import io
from PIL import Image

from models.cnn_encoder import CNNEncoder

app = FastAPI(title="Style Similarity API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the CNN encoder
encoder = CNNEncoder()

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    shape: List[int]

@app.get("/")
async def root():
    return {"message": "Style Similarity API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/encode", response_model=EmbeddingResponse)
async def encode_image(file: UploadFile = File(...)):
    """
    Encode an uploaded image into a feature vector using CNN
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Get embedding from CNN encoder
        embedding = encoder.encode(image)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            shape=list(embedding.shape)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/batch-encode")
async def batch_encode_images(files: List[UploadFile] = File(...)):
    """
    Encode multiple images in a batch
    """
    try:
        embeddings = []

        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            embedding = encoder.encode(image)
            embeddings.append({
                "filename": file.filename,
                "embedding": embedding.tolist(),
                "shape": list(embedding.shape)
            })

        return {"results": embeddings}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")

@app.post("/similarity")
async def compute_similarity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Compute cosine similarity between two images
    """
    try:
        # Process first image
        contents1 = await file1.read()
        image1 = Image.open(io.BytesIO(contents1)).convert('RGB')
        embedding1 = encoder.encode(image1)

        # Process second image
        contents2 = await file2.read()
        image2 = Image.open(io.BytesIO(contents2)).convert('RGB')
        embedding2 = encoder.encode(image2)

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        return {
            "similarity": float(similarity),
            "file1": file1.filename,
            "file2": file2.filename
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error computing similarity: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
