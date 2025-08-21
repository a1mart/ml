# models_router.py - FIXED VERSION
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import json
import inspect
import asyncio
from datetime import datetime
from src.domain.pmp import MLManager, TaskManager, TrainingStatus, task_manager

router = APIRouter(prefix="/models", tags=["Models"])
MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

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
    """FIXED: Combined training request model"""
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

@router.get("/available")
def available_models():
    """Return list of available models and their init parameters"""
    models_info = {}
    for name, cls in MLManager.AVAILABLE_MODELS.items():
        try:
            sig = inspect.signature(cls.__init__)
            # Skip 'self' and get parameter info
            params = {}
            for k, v in sig.parameters.items():
                if k != "self":
                    param_info = {
                        "default": str(v.default) if v.default != inspect.Parameter.empty else None,
                        "annotation": str(v.annotation) if v.annotation != inspect.Parameter.empty else None
                    }
                    params[k] = param_info
            models_info[name] = params
        except Exception as e:
            models_info[name] = {"error": f"Could not inspect model: {str(e)}"}
    
    return {"models": models_info}

@router.post("/register/{dataset_name}")
def register_model(dataset_name: str, model: ModelRegister):
    """Register a model for a specific dataset"""
    dataset_path = os.path.join("data/sets", dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    os.makedirs(dataset_model_dir, exist_ok=True)
    
    # Load existing registrations
    meta_path = os.path.join(dataset_model_dir, "registered_models.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {"models": {}, "dataset_info": {"target_column": None, "drop_columns": []}}
    
    # Update dataset info if provided
    meta["dataset_info"]["target_column"] = model.target_column
    meta["dataset_info"]["drop_columns"] = model.drop_columns or []
    
    # Register the model
    model_id = f"{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta["models"][model_id] = {
        "name": model.name,
        "model_name": model.model_name,
        "params": model.params or {},
        "registered_at": datetime.now().isoformat(),
        "status": "registered"
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    
    return {
        "status": "registered",
        "model_id": model_id,
        "model": model.name,
        "dataset": dataset_name
    }

async def train_model_background(task_id: str, dataset_name: str, model_ids: List[str], config: TrainingConfig):
    """Background task for training models"""
    try:
        # Update task status
        task_manager.tasks[task_id].status = TrainingStatus.RUNNING
        task_manager.tasks[task_id].start_time = datetime.now()
        
        dataset_path = os.path.join("data/sets", dataset_name)
        
        # Initialize ML Manager
        manager = MLManager(
            dataset_path=dataset_path,
            target_column=config.target_column,
            drop_columns=config.drop_columns,
            n_splits=config.n_splits,
            random_state=config.random_state,
            save_format=config.save_format,
            dataset_name=dataset_name
        )
        
        # Load registered models
        dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
        meta_path = os.path.join(dataset_model_dir, "registered_models.json")
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Register specified models
        for model_id in model_ids:
            if model_id in meta["models"]:
                model_info = meta["models"][model_id]
                manager.register_model(
                    model_info["name"],
                    model_info["model_name"],
                    **model_info.get("params", {})
                )
        
        # Progress callback
        async def progress_callback(progress, message):
            task_manager.update_task_progress(task_id, progress)
        
        # Train models
        await manager.train_all_async(progress_callback)
        
        # Collect results
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
        
        # Mark task as completed
        task_manager.complete_task(task_id, results)
        
    except Exception as e:
        task_manager.fail_task(task_id, str(e))

@router.post("/train/{dataset_name}", response_model=TrainingResponse)
async def train_models(
    dataset_name: str, 
    background_tasks: BackgroundTasks,
    request: TrainingRequest
):
    """FIXED: Start training models asynchronously with unified request structure"""
    dataset_path = os.path.join("data/sets", dataset_name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    meta_path = os.path.join(dataset_model_dir, "registered_models.json")
    
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="No models registered for this dataset")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Use model_ids from request body, or train all registered models if none specified
    model_ids = request.model_ids if request.model_ids else list(meta["models"].keys())
    
    # Validate model IDs
    invalid_models = [mid for mid in model_ids if mid not in meta["models"]]
    if invalid_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model IDs: {invalid_models}"
        )
    
    # Create training task
    task_id = task_manager.create_task(dataset_name, ",".join(model_ids))
    
    # Create training config from request
    config = TrainingConfig(
        target_column=request.target_column,
        drop_columns=request.drop_columns or [],
        n_splits=request.n_splits,
        random_state=request.random_state,
        save_format=request.save_format
    )
    
    # Start background training
    background_tasks.add_task(
        train_model_background, 
        task_id, 
        dataset_name, 
        model_ids, 
        config
    )
    
    return TrainingResponse(
        task_id=task_id,
        status="started",
        message=f"Training started for {len(model_ids)} models",
        poll_url=f"/models/tasks/{task_id}"
    )

@router.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str):
    """Get the status of a training task"""
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

@router.get("/list/{dataset_name}")
def list_models(dataset_name: str):
    """List all models (registered and trained) for a dataset"""
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    
    result = {
        "dataset": dataset_name,
        "registered_models": [],
        "trained_models": []
    }
    
    # Get registered models
    meta_path = os.path.join(dataset_model_dir, "registered_models.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            result["registered_models"] = meta.get("models", {})
    
    # Get trained models
    if os.path.exists(dataset_model_dir):
        try:
            manager = MLManager(
                dataset_path=os.path.join("data/sets", dataset_name),
                target_column="dummy",  # Will be overridden
                dataset_name=dataset_name
            )
            result["trained_models"] = manager.list_saved_models()
        except Exception as e:
            result["error"] = f"Could not load trained models: {str(e)}"
    
    return result

@router.delete("/trained/{dataset_name}/{model_id}")
def delete_trained_model(dataset_name: str, model_id: str):
    """Delete a trained model"""
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    model_files = [
        f"{model_id}.pkl",
        f"{model_id}.joblib", 
        f"{model_id}.onnx",
        f"{model_id}_report.json"
    ]
    
    deleted_files = []
    for filename in model_files:
        file_path = os.path.join(dataset_model_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(filename)
            except Exception as e:
                pass
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail="No model files found to delete")
    
    return {"status": "deleted", "files": deleted_files}