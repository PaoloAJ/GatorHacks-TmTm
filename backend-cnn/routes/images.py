"""
Image encoding and similarity routes
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import io
from PIL import Image

from schemas import EmbeddingResponse, SimilarityResponse, BatchEncodeResponse
from schemas.image_schemas import ImageEmbeddingResult
from services import EncoderService

router = APIRouter(prefix="/images", tags=["images"])

# Initialize encoder service
encoder_service = EncoderService()


@router.post("/encode", response_model=EmbeddingResponse)
async def encode_image(file: UploadFile = File(...)):
    """
    Encode an uploaded image into a feature vector using CNN.

    Args:
        file: Image file to encode

    Returns:
        EmbeddingResponse with embedding vector and shape
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Get embedding from encoder service
        embedding = encoder_service.encode_image(image)

        return EmbeddingResponse(
            embedding=embedding.tolist(),
            shape=list(embedding.shape)
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/batch-encode", response_model=BatchEncodeResponse)
async def batch_encode_images(files: List[UploadFile] = File(...)):
    """
    Encode multiple images in a batch.

    Args:
        files: List of image files to encode

    Returns:
        BatchEncodeResponse with list of embeddings
    """
    try:
        results = []

        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            embedding = encoder_service.encode_image(image)

            results.append(ImageEmbeddingResult(
                filename=file.filename,
                embedding=embedding.tolist(),
                shape=list(embedding.shape)
            ))

        return BatchEncodeResponse(results=results)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing images: {str(e)}"
        )


@router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Compute cosine similarity between two images.

    Args:
        file1: First image file
        file2: Second image file

    Returns:
        SimilarityResponse with similarity score and filenames
    """
    try:
        # Process first image
        contents1 = await file1.read()
        image1 = Image.open(io.BytesIO(contents1)).convert('RGB')
        embedding1 = encoder_service.encode_image(image1)

        # Process second image
        contents2 = await file2.read()
        image2 = Image.open(io.BytesIO(contents2)).convert('RGB')
        embedding2 = encoder_service.encode_image(image2)

        # Compute similarity using service
        similarity = encoder_service.compute_similarity(embedding1, embedding2)

        return SimilarityResponse(
            similarity=similarity,
            file1=file1.filename,
            file2=file2.filename
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error computing similarity: {str(e)}"
        )
