"""
Pydantic schemas for image-related API responses
"""
from pydantic import BaseModel
from typing import List


class EmbeddingResponse(BaseModel):
    """Response model for single image encoding"""
    embedding: List[float]
    shape: List[int]


class ImageEmbeddingResult(BaseModel):
    """Individual image result in batch encoding"""
    filename: str
    embedding: List[float]
    shape: List[int]


class BatchEncodeResponse(BaseModel):
    """Response model for batch image encoding"""
    results: List[ImageEmbeddingResult]


class SimilarityResponse(BaseModel):
    """Response model for similarity computation"""
    similarity: float
    file1: str
    file2: str
