from fastapi import FastAPI
from app.routers import test
from app.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Image Creditor")

app.include_router(test.router)

@app.get("/")
def read_root():
    return {"message": "Application running"}
