# inference_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pickle
import joblib
import numpy as np
from typing import List, Dict, Any

router = APIRouter(prefix="/inference", tags=["Inference"])
MODEL_DIR = "data/models"

class InferenceRequest(BaseModel):
    instances: List[List[float]]  # Input data for prediction
    model_id: str  # Unique model ID

class InferenceResponse(BaseModel):
    model_id: str
    dataset: str
    predictions: List[Any]
    prediction_probabilities: List[List[float]] = None
    feature_names: List[str] = None

class BatchInferenceRequest(BaseModel):
    dataset_file: str  # CSV file to predict on
    model_id: str
    output_column_name: str = "prediction"

@router.post("/{dataset_name}", response_model=InferenceResponse)
def inference(dataset_name: str, request: InferenceRequest):
    """Make predictions using a trained model"""
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    if not os.path.exists(dataset_model_dir):
        raise HTTPException(status_code=404, detail="No models found for this dataset")
    
    # Try loading model with different formats
    model_path_pkl = os.path.join(dataset_model_dir, f"{request.model_id}.pkl")
    model_path_joblib = os.path.join(dataset_model_dir, f"{request.model_id}.joblib")
    model = None
    
    try:
        if os.path.exists(model_path_pkl):
            with open(model_path_pkl, "rb") as f:
                model = pickle.load(f)
        elif os.path.exists(model_path_joblib):
            model = joblib.load(model_path_joblib)
        else:
            raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Load model metadata
    report_path = os.path.join(dataset_model_dir, f"{request.model_id}_report.json")
    feature_names = None
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = json.load(f)
            feature_names = report.get("feature_names", [])
    
    try:
        # Validate input dimensions
        if feature_names and len(request.instances[0]) != len(feature_names):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(feature_names)} features, got {len(request.instances[0])}"
            )
        
        X = np.array(request.instances)
        predictions = model.predict(X)
        
        # Try to get prediction probabilities
        prediction_probabilities = None
        try:
            if hasattr(model, "predict_proba"):
                prediction_probabilities = model.predict_proba(X).tolist()
        except:
            pass  # Some models don't support probability prediction
        
        return InferenceResponse(
            model_id=request.model_id,
            dataset=dataset_name,
            predictions=predictions.tolist(),
            prediction_probabilities=prediction_probabilities,
            feature_names=feature_names
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/{dataset_name}/batch")
def batch_inference(dataset_name: str, request: BatchInferenceRequest):
    """Make batch predictions on a CSV file"""
    # Load the CSV file
    input_file = os.path.join("data/sets", request.dataset_file)
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Input dataset file not found")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    # Load model
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    model_path_pkl = os.path.join(dataset_model_dir, f"{request.model_id}.pkl")
    model_path_joblib = os.path.join(dataset_model_dir, f"{request.model_id}.joblib")
    
    try:
        if os.path.exists(model_path_pkl):
            with open(model_path_pkl, "rb") as f:
                model = pickle.load(f)
        elif os.path.exists(model_path_joblib):
            model = joblib.load(model_path_joblib)
        else:
            raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    try:
        # Make predictions
        predictions = model.predict(df.values)
        
        # Add predictions to dataframe
        df[request.output_column_name] = predictions
        
        # Save results
        output_file = f"predictions_{dataset_name}_{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join("data/sets", output_file)
        df.to_csv(output_path, index=False)
        
        return {
            "status": "completed",
            "input_file": request.dataset_file,
            "output_file": output_file,
            "predictions_count": len(predictions),
            "output_path": output_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.get("/{dataset_name}/models")
def list_available_models_for_inference(dataset_name: str):
    """List all trained models available for inference"""
    dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
    if not os.path.exists(dataset_model_dir):
        raise HTTPException(status_code=404, detail="No models found for this dataset")
    
    available_models = []
    for file in os.listdir(dataset_model_dir):
        if file.endswith(('.pkl', '.joblib')):
            model_id = file.rsplit('.', 1)[0]
            report_path = os.path.join(dataset_model_dir, f"{model_id}_report.json")
            
            model_info = {
                "model_id": model_id,
                "format": file.split('.')[-1],
                "file_path": os.path.join(dataset_model_dir, file),
                "has_report": os.path.exists(report_path)
            }
            
            if model_info["has_report"]:
                try:
                    with open(report_path, "r") as f:
                        report = json.load(f)
                        model_info.update({
                            "model_name": report.get("model_name", "Unknown"),
                            "f1_score": report.get("mean_f1", 0),
                            "feature_count": report.get("n_features", 0),
                            "timestamp": report.get("timestamp", "Unknown")
                        })
                except:
                    pass
            
            available_models.append(model_info)
    
    return {
        "dataset": dataset_name,
        "available_models": sorted(available_models, key=lambda x: x.get("f1_score", 0), reverse=True)
    }

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