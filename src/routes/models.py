from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os, json, inspect

from src.domain.pmp import MLManager

router = APIRouter(prefix="/models", tags=["Models"])

MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelRegister(BaseModel):
    name: str
    model_name: str
    params: Optional[dict] = {}

@router.get("/available")
def available_models():
    """Return list of available models and their init parameters"""
    models_info = {}
    for name, cls in MLManager.AVAILABLE_MODELS.items():
        sig = inspect.signature(cls.__init__)
        # Skip 'self'
        params = {k: str(v.default) for k, v in sig.parameters.items() if k != "self"}
        models_info[name] = params
    return {"models": models_info}

@router.post("/register/{dataset_name}")
def register_model(dataset_name: str, model: ModelRegister):
    dataset_path = os.path.join("data/sets", dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    os.makedirs(dataset_model_dir, exist_ok=True)

    meta_path = os.path.join(dataset_model_dir, "models.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta[model.name] = {"model_name": model.model_name, "params": model.params or {}}

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    return {"status": "registered", "model": model.name}

@router.post("/train/{dataset_name}/{model_name}")
def train_model(dataset_name: str, model_name: str):
    dataset_path = os.path.join("data/sets", dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    meta_path = os.path.join(dataset_model_dir, "models.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="No models registered")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    if model_name not in meta:
        raise HTTPException(status_code=404, detail="Model not registered")

    model_info = meta[model_name]
    manager = MLManager(dataset_path=dataset_path, target_column="Class")
    manager.register_model(model_name, model_info["model_name"], **model_info.get("params", {}))
    manager.train_all()

    # Save trained model & report
    for score, name, pipe, report in manager.rank_models():
        if name == model_name:
            model_file = os.path.join(dataset_model_dir, f"{name}.{manager.save_format}")
            report_file = os.path.join(dataset_model_dir, f"{name}_report.json")
            manager._save_model_and_report(pipe, name, report)
            break

    return {"status": "trained", "model": model_name}

@router.get("/list/{dataset_name}")
def list_models(dataset_name: str):
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    if not os.path.exists(dataset_model_dir):
        raise HTTPException(status_code=404, detail="No models found for this dataset")
    models = []
    for file in os.listdir(dataset_model_dir):
        if file.endswith(".pkl") or file.endswith(".joblib") or file.endswith(".onnx"):
            report_file = os.path.join(dataset_model_dir, file.replace(".pkl","_report.json").replace(".joblib","_report.json").replace(".onnx","_report.json"))
            models.append({
                "model_file": file,
                "report_available": os.path.exists(report_file)
            })
    return {"models": models}
