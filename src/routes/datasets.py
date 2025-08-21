from fastapi import APIRouter, UploadFile, File, HTTPException
import os, shutil
import pandas as pd

router = APIRouter(prefix="/datasets", tags=["Datasets"])

DATA_DIR = "data/sets"
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "status": "uploaded"}

@router.get("/")
def list_datasets():
    return {"datasets": os.listdir(DATA_DIR)}

@router.get("/{dataset_name}")
def get_dataset_info(dataset_name: str):
    file_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_csv(file_path)
    return {
        "columns": list(df.columns),
        "rows": len(df),
    }
