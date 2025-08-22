
# testing_workflow.py - Complete workflow testing
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import tempfile
import shutil
from typing import List, Dict, Any

router = APIRouter(prefix="/testing", tags=["Testing Workflow"])

class WorkflowTest(BaseModel):
    dataset_name: str = "test_dataset"
    n_samples: int = 1000
    n_features: int = 20
    n_classes: int = 2
    models_to_test: List[str] = ["LogisticRegression", "RandomForest", "XGBoost"]

@router.post("/complete-workflow")
async def test_complete_workflow(test_config: WorkflowTest):
    """Test the complete ML workflow from dataset creation to inference"""
    
    workflow_results = {
        "steps": [],
        "errors": [],
        "final_status": "in_progress"
    }
    
    try:
        # Step 1: Generate synthetic dataset
        X, y = make_classification(
            n_samples=test_config.n_samples,
            n_features=test_config.n_features,
            n_classes=test_config.n_classes,
            n_informative=test_config.n_features // 2,
            n_redundant=test_config.n_features // 4,
            random_state=42
        )
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(test_config.n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save dataset
        dataset_path = os.path.join("data/sets", f"{test_config.dataset_name}.csv")
        df.to_csv(dataset_path, index=False)
        
        workflow_results["steps"].append({
            "step": "dataset_creation",
            "status": "success",
            "details": f"Created dataset with {test_config.n_samples} samples and {test_config.n_features} features"
        })
        
        # Step 2: Register models
        from models_router import register_model, ModelRegister
        
        model_configs = {
            "LogisticRegression": {"max_iter": 1000, "random_state": 42},
            "RandomForest": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
            "XGBoost": {"n_estimators": 50, "max_depth": 3, "random_state": 42}
        }
        
        registered_models = []
        for model_name in test_config.models_to_test:
            if model_name in model_configs:
                try:
                    model_reg = ModelRegister(
                        name=f"{model_name}_test",
                        model_name=model_name,
                        params=model_configs[model_name],
                        target_column="target",
                        drop_columns=[]
                    )
                    
                    result = register_model(test_config.dataset_name, model_reg)
                    registered_models.append(result["model_id"])
                    
                except Exception as e:
                    workflow_results["errors"].append(f"Error registering {model_name}: {str(e)}")
        
        workflow_results["steps"].append({
            "step": "model_registration",
            "status": "success",
            "details": f"Registered {len(registered_models)} models: {registered_models}"
        })
        
        # Step 3: Train models (simulate async training)
        from models_router import TrainingConfig, train_model_background
        from enhanced_ml_framework import task_manager
        
        config = TrainingConfig(
            target_column="target",
            drop_columns=[],
            n_splits=3,  # Reduced for faster testing
            random_state=42,
            save_format="pkl"
        )
        
        # Create and run training task
        task_id = task_manager.create_task(test_config.dataset_name, ",".join(registered_models))
        await train_model_background(task_id, test_config.dataset_name, registered_models, config)
        
        # Get training results
        task = task_manager.get_task(task_id)
        if task.status.value == "completed":
            workflow_results["steps"].append({
                "step": "model_training",
                "status": "success",
                "details": f"Trained {task.results['models_trained']} models",
                "results": task.results
            })
        else:
            workflow_results["errors"].append(f"Training failed: {task.error_message}")
        
        # Step 4: Test inference
        if task.results and task.results["saved_models"]:
            best_model = task.results["saved_models"][0]  # Get best model
            model_id = best_model["unique_id"]
            
            # Create test data for inference
            test_X = X[:5]  # Use first 5 samples for testing
            
            try:
                from inference_router import inference, InferenceRequest
                
                inf_request = InferenceRequest(
                    instances=test_X.tolist(),
                    model_id=model_id
                )
                
                predictions = inference(test_config.dataset_name, inf_request)
                
                workflow_results["steps"].append({
                    "step": "inference",
                    "status": "success",
                    "details": f"Made {len(predictions.predictions)} predictions",
                    "predictions": predictions.predictions
                })
                
            except Exception as e:
                workflow_results["errors"].append(f"Inference error: {str(e)}")
        
        # Final status
        if not workflow_results["errors"]:
            workflow_results["final_status"] = "success"
        else:
            workflow_results["final_status"] = "partial_success"
        
    except Exception as e:
        workflow_results["errors"].append(f"Workflow error: {str(e)}")
        workflow_results["final_status"] = "failed"
    
    return workflow_results

@router.get("/cleanup/{dataset_name}")
def cleanup_test_data(dataset_name: str):
    """Clean up test data and models"""
    cleanup_results = {
        "deleted_files": [],
        "errors": []
    }
    
    # Delete dataset
    dataset_path = os.path.join("data/sets", f"{dataset_name}.csv")
    if os.path.exists(dataset_path):
        try:
            os.remove(dataset_path)
            cleanup_results["deleted_files"].append(dataset_path)
        except Exception as e:
            cleanup_results["errors"].append(f"Error deleting dataset: {str(e)}")
    
    # Delete model directory
    model_dir = os.path.join("data/models", dataset_name)
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            cleanup_results["deleted_files"].append(model_dir)
        except Exception as e:
            cleanup_results["errors"].append(f"Error deleting model directory: {str(e)}")
    
    return cleanup_results