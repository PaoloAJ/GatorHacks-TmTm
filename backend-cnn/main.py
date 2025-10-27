"""
Style Similarity API - Main Application

A FastAPI application for image style similarity analysis using CNN encoders.
Organized with clean separation of concerns:
- Routes: API endpoints
- Services: Business logic
- Models: ML models
- Schemas: Request/Response models
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import images_router

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="API for encoding images and computing style similarity using CNN"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(images_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": settings.app_name,
        "version": settings.version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
