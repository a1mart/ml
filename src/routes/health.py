from fastapi import APIRouter
from datetime import datetime

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/")
def root():
    return {
        "message": "ML Framework API",
        "version": "0.0.0",
        "endpoints": {
            "datasets": "/docs#/Datasets",
            "models": "/docs#/Models", 
            "inference": "/docs#/Inference",
            "testing": "/docs#/Testing%20Workflow"
        }
    }