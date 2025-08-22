from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List

import pandas as pd

from src.deps.storage import get_storage
from src.storage.base import BaseStorage

router = APIRouter(prefix="/datasets", tags=["Datasets"])


class DatasetInfo(BaseModel):
    filename: str
    columns: List[str]
    rows: int
    size_bytes: int
    file_type: str
    target_suggestions: List[str]


@router.post("/upload/", response_model=dict)
async def upload_dataset(
    file: UploadFile = File(...),
    storage: BaseStorage = Depends(get_storage)
):
    """Upload a dataset file using the configured storage backend"""
    if not file.filename.endswith((".csv", ".json")):
        raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported")

    data = await file.read()

    try:
        # Save file via storage backend
        path = storage.save_file(file.filename, data)
        size = storage.get_file_size(file.filename)
        return {
            "filename": file.filename,
            "size_bytes": size,
            "path": path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@router.get("/", response_model=dict)
def list_datasets(storage: BaseStorage = Depends(get_storage)):
    """List all uploaded datasets"""
    try:
        files = storage.list_files()
        datasets = [
            {"filename": f, "size_bytes": storage.get_file_size(f), "path": f} for f in files
        ]
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@router.get("/{dataset_name}", response_model=DatasetInfo)
def get_dataset_info(dataset_name: str, storage: BaseStorage = Depends(get_storage)):
    """Get detailed information about a dataset"""
    if not storage.file_exists(dataset_name):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        data = storage.get_file(dataset_name)
        df = None
        if dataset_name.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(data))
            file_type = "CSV"
        elif dataset_name.endswith(".json"):
            df = pd.read_json(pd.io.common.BytesIO(data))
            file_type = "JSON"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Suggest potential target columns (binary, categorical with few unique values)
        target_suggestions = [
            col for col in df.columns
            if df[col].nunique() == 2 or (df[col].nunique() <= 10 and df[col].dtype in ["object", "int64"])
        ]

        size = storage.get_file_size(dataset_name)

        return DatasetInfo(
            filename=dataset_name,
            columns=list(df.columns),
            rows=len(df),
            size_bytes=size,
            file_type=file_type,
            target_suggestions=target_suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")


@router.delete("/{dataset_name}")
def delete_dataset(dataset_name: str, storage: BaseStorage = Depends(get_storage)):
    """Delete a dataset"""
    if not storage.file_exists(dataset_name):
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        storage.delete_file(dataset_name)
        return {"status": "deleted", "filename": dataset_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")
