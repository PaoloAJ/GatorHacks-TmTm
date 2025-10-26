"""
Image encoding and similarity routes
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import io
from PIL import Image

from schemas import EmbeddingResponse, SimilarityResponse, BatchEncodeResponse
from schemas.image_schemas import ImageEmbeddingResult, SimilarArtistResponse
from services import EncoderService
from services.pinecone_service import PineconeService
from services.artist_style_service import ArtistStyleService

router = APIRouter(prefix="/images", tags=["images"])

# Initialize services
encoder_service = EncoderService()
pinecone_service = PineconeService()
artist_style_service = ArtistStyleService()


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

        # Handle potential NaN or inf values
        if not (-1.0 <= similarity <= 1.0):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid similarity score: {similarity}"
            )

        return SimilarityResponse(
            similarity=float(similarity),
            file1=file1.filename or "unknown",
            file2=file2.filename or "unknown"
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error computing similarity: {str(e)}"
        )


@router.post("/find-similar-artist", response_model=SimilarArtistResponse)
async def find_similar_artist(
    file: UploadFile = File(...),
    top_k: int = 10
):
    """
    Find artists with similar style to the uploaded image.

    This endpoint:
    1. Encodes the uploaded image
    2. Queries Pinecone for similar embeddings
    3. Returns top K most similar artists

    Args:
        file: Image file to analyze
        top_k: Number of similar artists to return (default: 10)

    Returns:
        SimilarArtistResponse with list of similar artists
    """
    try:
        # Encode the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        embedding = encoder_service.encode_image(image)

        # Query Pinecone for similar images
        results = pinecone_service.query_similar(
            query_embedding=embedding,
            top_k=top_k
        )

        # Format results
        matches = []
        for result in results:
            metadata = result['metadata']
            matches.append({
                'artist_name': metadata.get('artist_name', 'Unknown'),
                'similarity_score': result['score'],
                'image_filename': metadata.get('image_filename', ''),
                'image_path': metadata.get('image_path', '')
            })

        return SimilarArtistResponse(
            query_filename=file.filename or "unknown",
            matches=matches,
            total_results=len(matches)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar artists: {str(e)}"
        )


@router.post("/match-artist-style")
async def match_artist_style(
    file: UploadFile = File(...),
    top_k: int = 5,
    sample_size: int = 50
):
    """
    Match uploaded image to artist styles (not individual artworks).

    This endpoint finds which artists' overall style best matches the uploaded image
    by aggregating similarity scores across multiple artworks.

    Args:
        file: Image file to analyze
        top_k: Number of top matching artists to return (default: 5)
        sample_size: Number of similar images to sample for aggregation (default: 50)

    Returns:
        Dictionary with top matching artists and their aggregated scores
    """
    try:
        # Encode the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        embedding = encoder_service.encode_image(image)

        # Find matching artists using aggregation strategy
        artist_matches = artist_style_service.find_artist_by_aggregation(
            query_embedding=embedding,
            top_k=top_k,
            sample_size=sample_size
        )

        return {
            'query_filename': file.filename or "unknown",
            'matching_artists': artist_matches,
            'total_artists_found': len(artist_matches),
            'sample_size': sample_size
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error matching artist style: {str(e)}"
        )


@router.post("/match-artist-style-fast")
async def match_artist_style_fast(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Fast artist style matching using pre-computed centroids.

    This endpoint uses pre-computed "hero" style vectors for each artist,
    providing much faster results than the aggregation method.

    Requires artist centroids to be uploaded to Pinecone first
    (run upload script with centroid computation).

    Args:
        file: Image file to analyze
        top_k: Number of top matching artists to return (default: 5)

    Returns:
        Dictionary with top matching artists based on centroid similarity
    """
    try:
        # Encode the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        embedding = encoder_service.encode_image(image)

        # Find matching artists using centroid method
        artist_matches = artist_style_service.find_artist_by_centroid(
            query_embedding=embedding,
            top_k=top_k
        )

        return {
            'query_filename': file.filename or "unknown",
            'matching_artists': artist_matches,
            'total_artists_found': len(artist_matches),
            'method': 'centroid'
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error matching artist style: {str(e)}"
        )
