from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Any
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.deps.storage import get_storage
from src.storage.base import BaseStorage

router = APIRouter(prefix="/inference", tags=["Inference"])


# ----------------------
# Pydantic Models
# ----------------------
class InferenceRequest(BaseModel):
    instances: List[List[float]]
    model_id: str


class InferenceResponse(BaseModel):
    model_id: str
    dataset: str
    predictions: List[Any]
    prediction_probabilities: List[List[float]] = None
    feature_names: List[str] = None


class BatchInferenceRequest(BaseModel):
    dataset_file: str
    model_id: str
    output_column_name: str = "prediction"


# ----------------------
# Single inference
# ----------------------
@router.post("/{dataset_name}", response_model=InferenceResponse)
def inference(dataset_name: str, request: InferenceRequest, storage: BaseStorage = Depends(get_storage)):
    """Make predictions using a trained model"""
    dataset_model_dir = f"models/{dataset_name}"
    if not storage.dir_exists(dataset_model_dir):
        raise HTTPException(status_code=404, detail="No models found for this dataset")

    # Load model
    model = None
    for ext in ["pkl", "joblib"]:
        model_path = f"{dataset_model_dir}/{request.model_id}.{ext}"
        if storage.file_exists(model_path):
            with storage.open_file(model_path, "rb") as f:
                model = pickle.load(f) if ext == "pkl" else joblib.load(f)
            break
    if not model:
        raise HTTPException(status_code=404, detail="Model file not found")

    # Load metadata
    report_path = f"{dataset_model_dir}/{request.model_id}_report.json"
    feature_names = None
    if storage.file_exists(report_path):
        report = json.loads(storage.get_file(report_path).decode())
        feature_names = report.get("feature_names", [])

    # Validate input
    if feature_names and len(request.instances[0]) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(request.instances[0])}"
        )

    try:
        X = np.array(request.instances)
        predictions = model.predict(X)

        prediction_probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                prediction_probabilities = model.predict_proba(X).tolist()
            except Exception:
                pass

        return InferenceResponse(
            model_id=request.model_id,
            dataset=dataset_name,
            predictions=predictions.tolist(),
            prediction_probabilities=prediction_probabilities,
            feature_names=feature_names
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ----------------------
# Batch inference
# ----------------------
@router.post("/{dataset_name}/batch")
def batch_inference(dataset_name: str, request: BatchInferenceRequest, storage: BaseStorage = Depends(get_storage)):
    """Make batch predictions on a CSV file"""
    input_file = f"datasets/{request.dataset_file}"
    if not storage.file_exists(input_file):
        raise HTTPException(status_code=404, detail="Input dataset file not found")

    try:
        df_bytes = storage.get_file(input_file)
        df = pd.read_csv(pd.io.common.BytesIO(df_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    # Load model
    dataset_model_dir = f"models/{dataset_name}"
    model = None
    for ext in ["pkl", "joblib"]:
        model_path = f"{dataset_model_dir}/{request.model_id}.{ext}"
        if storage.file_exists(model_path):
            with storage.open_file(model_path, "rb") as f:
                model = pickle.load(f) if ext == "pkl" else joblib.load(f)
            break
    if not model:
        raise HTTPException(status_code=404, detail="Model file not found")

    # Make predictions
    try:
        predictions = model.predict(df.values)
        df[request.output_column_name] = predictions

        output_file = f"predictions_{dataset_name}_{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        storage.save_file(f"datasets/{output_file}", df.to_csv(index=False).encode())

        return {
            "status": "completed",
            "input_file": request.dataset_file,
            "output_file": output_file,
            "predictions_count": len(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ----------------------
# List available models for inference
# ----------------------
@router.get("/{dataset_name}/models")
def list_available_models_for_inference(dataset_name: str, storage: BaseStorage = Depends(get_storage)):
    dataset_model_dir = f"models/{dataset_name}"
    if not storage.dir_exists(dataset_model_dir):
        raise HTTPException(status_code=404, detail="No models found for this dataset")

    available_models = []
    for file in storage.list_dir(dataset_model_dir):
        if file.endswith((".pkl", ".joblib")):
            model_id = file.rsplit(".", 1)[0]
            report_path = f"{dataset_model_dir}/{model_id}_report.json"

            model_info = {"model_id": model_id, "format": file.split(".")[-1], "has_report": storage.file_exists(report_path)}
            if model_info["has_report"]:
                try:
                    report = json.loads(storage.get_file(report_path).decode())
                    model_info.update({
                        "model_name": report.get("model_name", "Unknown"),
                        "f1_score": report.get("mean_f1", 0),
                        "feature_count": report.get("n_features", 0),
                        "timestamp": report.get("timestamp", "Unknown")
                    })
                except Exception:
                    pass
            available_models.append(model_info)

    return {
        "dataset": dataset_name,
        "available_models": sorted(available_models, key=lambda x: x.get("f1_score", 0), reverse=True)
    }
