# datasets_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import os
import shutil
import pandas as pd
from typing import List, Optional

router = APIRouter(prefix="/datasets", tags=["Datasets"])
DATA_DIR = "data/sets"
os.makedirs(DATA_DIR, exist_ok=True)

class DatasetInfo(BaseModel):
    filename: str
    columns: List[str]
    rows: int
    size_bytes: int
    file_type: str
    target_suggestions: List[str]

@router.post("/upload/", response_model=dict)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file"""
    if not file.filename.endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported")
    
    # Create safe filename
    safe_filename = file.filename.replace(" ", "_").replace("/", "_")
    file_path = os.path.join(DATA_DIR, safe_filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        base_name, ext = os.path.splitext(safe_filename)
        counter = 1
        while os.path.exists(file_path):
            safe_filename = f"{base_name}_{counter}{ext}"
            file_path = os.path.join(DATA_DIR, safe_filename)
            counter += 1
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return {
            "filename": safe_filename,
            "original_filename": file.filename,
            "status": "uploaded",
            "size_bytes": file_size,
            "path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/", response_model=dict)
def list_datasets():
    """List all uploaded datasets"""
    try:
        datasets = []
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                datasets.append({
                    "filename": filename,
                    "size_bytes": size,
                    "path": file_path
                })
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@router.get("/{dataset_name}", response_model=DatasetInfo)
def get_dataset_info(dataset_name: str):
    """Get detailed information about a dataset"""
    file_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        if dataset_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif dataset_name.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Suggest potential target columns (binary, categorical with few unique values)
        target_suggestions = []
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2:  # Binary column
                target_suggestions.append(col)
            elif unique_vals <= 10 and df[col].dtype in ['object', 'int64']:  # Categorical
                target_suggestions.append(col)
        
        file_size = os.path.getsize(file_path)
        file_type = 'CSV' if dataset_name.endswith('.csv') else 'JSON'
        
        return DatasetInfo(
            filename=dataset_name,
            columns=list(df.columns),
            rows=len(df),
            size_bytes=file_size,
            file_type=file_type,
            target_suggestions=target_suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

@router.delete("/{dataset_name}")
def delete_dataset(dataset_name: str):
    """Delete a dataset"""
    file_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        os.remove(file_path)
        return {"status": "deleted", "filename": dataset_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")

