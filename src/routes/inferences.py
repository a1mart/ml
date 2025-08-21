from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle
import joblib

router = APIRouter(prefix="/inference", tags=["Inference"])

MODEL_DIR = "data/models"

class InferenceRequest(BaseModel):
    instances: list[list[float]]  # new rows for prediction
    model_name: str

@router.post("/{dataset_name}")
def inference(dataset_name: str, request: InferenceRequest):
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    if not os.path.exists(dataset_model_dir):
        raise HTTPException(status_code=404, detail="No models found for this dataset")

    # Try loading model
    model_path_pkl = os.path.join(dataset_model_dir, f"{request.model_name}.pkl")
    model_path_joblib = os.path.join(dataset_model_dir, f"{request.model_name}.joblib")

    if os.path.exists(model_path_pkl):
        with open(model_path_pkl, "rb") as f:
            model = pickle.load(f)
    elif os.path.exists(model_path_joblib):
        model = joblib.load(model_path_joblib)
    else:
        raise HTTPException(status_code=404, detail="Model file not found")

    X = pd.DataFrame(request.instances).values
    y_pred = model.predict(X)

    return {
        "model": request.model_name,
        "dataset": dataset_name,
        "predictions": y_pred.tolist()
    }
