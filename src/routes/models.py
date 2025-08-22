from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import inspect
import asyncio

from src.deps.storage import get_storage
from src.storage.base import BaseStorage
from src.domain.pmp import MLManager, TaskManager, TrainingStatus, task_manager

router = APIRouter(prefix="/models", tags=["Models"])


# ----------------------
# Pydantic Models
# ----------------------
class ModelRegister(BaseModel):
    name: str
    model_name: str
    params: Optional[Dict[str, Any]] = {}
    target_column: str
    drop_columns: Optional[List[str]] = []


class TrainingConfig(BaseModel):
    target_column: str
    drop_columns: Optional[List[str]] = []
    n_splits: int = 5
    random_state: int = 1
    save_format: str = "pkl"


class TrainingRequest(BaseModel):
    target_column: str
    drop_columns: Optional[List[str]] = []
    n_splits: int = 5
    random_state: int = 1
    save_format: str = "pkl"
    model_ids: Optional[List[str]] = None


class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str
    poll_url: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    results: Optional[Dict] = None


# ----------------------
# Available models
# ----------------------
@router.get("/available")
def available_models():
    """Return list of available models and their init parameters"""
    models_info = {}
    for name, cls in MLManager.AVAILABLE_MODELS.items():
        try:
            sig = inspect.signature(cls.__init__)
            params = {
                k: {
                    "default": str(v.default) if v.default != inspect.Parameter.empty else None,
                    "annotation": str(v.annotation) if v.annotation != inspect.Parameter.empty else None
                }
                for k, v in sig.parameters.items() if k != "self"
            }
            models_info[name] = params
        except Exception as e:
            models_info[name] = {"error": f"Could not inspect model: {str(e)}"}
    return {"models": models_info}


# ----------------------
# Register a model
# ----------------------
@router.post("/register/{dataset_name}")
def register_model(
    dataset_name: str,
    model: ModelRegister,
    storage: BaseStorage = Depends(get_storage)
):
    """Register a model for a dataset"""
    if not storage.file_exists(dataset_name):
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_model_dir = f"models/{dataset_name}"
    storage.ensure_dir(dataset_model_dir)

    meta_path = f"{dataset_model_dir}/registered_models.json"
    meta = {}
    if storage.file_exists(meta_path):
        meta = json.loads(storage.get_file(meta_path).decode())
    else:
        meta = {"models": {}, "dataset_info": {"target_column": None, "drop_columns": []}}

    meta["dataset_info"]["target_column"] = model.target_column
    meta["dataset_info"]["drop_columns"] = model.drop_columns or []

    model_id = f"{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta["models"][model_id] = {
        "name": model.name,
        "model_name": model.model_name,
        "params": model.params or {},
        "registered_at": datetime.now().isoformat(),
        "status": "registered"
    }

    storage.save_file(meta_path, json.dumps(meta).encode())

    return {
        "status": "registered",
        "model_id": model_id,
        "model": model.name,
        "dataset": dataset_name
    }


# ----------------------
# Train models (background)
# ----------------------
async def train_model_background(
    task_id: str,
    dataset_name: str,
    model_ids: List[str],
    config: TrainingConfig,
    storage: BaseStorage
):
    try:
        task_manager.tasks[task_id].status = TrainingStatus.RUNNING
        task_manager.tasks[task_id].start_time = datetime.now()

        storage = get_storage()
        manager = MLManager(
            dataset_path=dataset_name,
            target_column=config.target_column,
            drop_columns=config.drop_columns,
            n_splits=config.n_splits,
            random_state=config.random_state,
            save_format=config.save_format,
            storage=storage,
            dataset_name=dataset_name
        )

        dataset_model_dir = f"models/{dataset_name}"
        meta_path = f"{dataset_model_dir}/registered_models.json"
        meta = json.loads(storage.get_file(meta_path).decode())

        for model_id in model_ids:
            if model_id in meta["models"]:
                info = meta["models"][model_id]
                manager.register_model(info["name"], info["model_name"], **info.get("params", {}))

        async def progress_callback(progress, message):
            task_manager.update_task_progress(task_id, progress)

        await manager.train_all_async(progress_callback)

        results = {
            "rankings": [],
            "models_trained": len(manager.results),
            "saved_models": manager.list_saved_models()
        }
        for score, name, _, report, unique_id in manager.rank_models():
            results["rankings"].append({
                "name": name,
                "unique_id": unique_id,
                "f1_score": score,
                "std_f1": report.get("std_f1", 0),
                "cv_scores": report.get("cv_scores", [])
            })

        task_manager.complete_task(task_id, results)
    except Exception as e:
        task_manager.fail_task(task_id, str(e))


@router.post("/train/{dataset_name}", response_model=TrainingResponse)
async def train_models(
    dataset_name: str,
    background_tasks: BackgroundTasks,
    request: TrainingRequest,
    storage: BaseStorage = Depends(get_storage)
):
    """Start async training for registered models"""
    if not storage.file_exists(dataset_name):
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_model_dir = f"models/{dataset_name}"
    meta_path = f"{dataset_model_dir}/registered_models.json"
    if not storage.file_exists(meta_path):
        raise HTTPException(status_code=404, detail="No models registered for this dataset")

    meta = json.loads(storage.get_file(meta_path).decode())
    model_ids = request.model_ids or list(meta["models"].keys())

    invalid_models = [mid for mid in model_ids if mid not in meta["models"]]
    if invalid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model IDs: {invalid_models}")

    task_id = task_manager.create_task(dataset_name, ",".join(model_ids))

    config = TrainingConfig(
        target_column=request.target_column,
        drop_columns=request.drop_columns or [],
        n_splits=request.n_splits,
        random_state=request.random_state,
        save_format=request.save_format
    )

    background_tasks.add_task(train_model_background, task_id, dataset_name, model_ids, config, storage)

    return TrainingResponse(
        task_id=task_id,
        status="started",
        message=f"Training started for {len(model_ids)} models",
        poll_url=f"/models/tasks/{task_id}"
    )


# ----------------------
# Task status
# ----------------------
@router.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatus(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        start_time=task.start_time.isoformat() if task.start_time else None,
        end_time=task.end_time.isoformat() if task.end_time else None,
        error_message=task.error_message,
        results=task.results
    )


# ----------------------
# List registered and trained models
# ----------------------
@router.get("/list/{dataset_name}")
def list_models(dataset_name: str, storage: BaseStorage = Depends(get_storage)):
    dataset_model_dir = f"models/{dataset_name}"
    result = {"dataset": dataset_name, "registered_models": [], "trained_models": []}

    meta_path = f"{dataset_model_dir}/registered_models.json"
    if storage.file_exists(meta_path):
        meta = json.loads(storage.get_file(meta_path).decode())
        result["registered_models"] = meta.get("models", {})

    try:
        manager = MLManager(dataset_path=dataset_name, target_column="dummy", storage=get_storage, dataset_name=dataset_name)
        result["trained_models"] = manager.list_saved_models()
    except Exception as e:
        result["error"] = f"Could not load trained models: {str(e)}"

    return result


# ----------------------
# Delete trained model
# ----------------------
@router.delete("/trained/{dataset_name}/{model_id}")
def delete_trained_model(dataset_name: str, model_id: str, storage: BaseStorage = Depends(get_storage)):
    dataset_model_dir = f"models/{dataset_name}"
    model_files = [f"{model_id}.pkl", f"{model_id}.joblib", f"{model_id}.onnx", f"{model_id}_report.json"]

    deleted_files = []
    for f in model_files:
        path = f"{dataset_model_dir}/{f}"
        if storage.file_exists(path):
            storage.delete_file(path)
            deleted_files.append(f)

    if not deleted_files:
        raise HTTPException(status_code=404, detail="No model files found to delete")

    return {"status": "deleted", "files": deleted_files}
