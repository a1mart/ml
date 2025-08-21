import os
import pickle
import joblib
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Available ML models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TrainingTask:
    task_id: str
    dataset_name: str
    model_name: str
    status: TrainingStatus
    progress: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict] = None

class MLManager:
    AVAILABLE_MODELS = {
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "Lasso": Lasso,
        "KNN": KNeighborsClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier,
        "SVC": SVC,
        "GaussianNB": GaussianNB,
        "XGBoost": XGBClassifier,
    }
    
    if LIGHTGBM_AVAILABLE:
        AVAILABLE_MODELS["LightGBM"] = LGBMClassifier
    
    if CATBOOST_AVAILABLE:
        AVAILABLE_MODELS["CatBoost"] = CatBoostClassifier

    SUPPORTED_FORMATS = ["pkl", "joblib", "onnx", "json"]

    def __init__(self, dataset_path, target_column, drop_columns=None, random_state=1, n_splits=5,
                 save_dir="data/models", save_format="pkl", dataset_name=None):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.drop_columns = drop_columns or []
        self.random_state = random_state
        self.n_splits = n_splits
        self.save_format = save_format.lower()
        self.dataset_name = dataset_name or os.path.basename(dataset_path).split('.')[0]
        
        # Create unique model directory for this dataset
        self.save_dir = os.path.join(save_dir, self.dataset_name)
        
        if self.save_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported save_format: {self.save_format}. Supported: {self.SUPPORTED_FORMATS}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.models = {}
        self.results = []
        self._load_dataset()

    def _load_dataset(self):
        """Load and preprocess the dataset"""
        try:
            if self.dataset_path.endswith('.csv'):
                data = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith('.json'):
                data = pd.read_json(self.dataset_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            if self.drop_columns:
                data = data.drop(columns=[c for c in self.drop_columns if c in data.columns])
            data = data.drop_duplicates()
            
            # Handle missing values
            data = data.dropna()
            
            self.X = data.drop(columns=[self.target_column]).values
            self.y = data[self.target_column].values
            self.feature_names = list(data.drop(columns=[self.target_column]).columns)
            
        except Exception as e:
            raise ValueError(f"Error loading dataset: {str(e)}")

    def register_model(self, name, model_name, **params):
        """Register a model with specific parameters"""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' is not available. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        # Create unique model name if already exists
        original_name = name
        counter = 1
        while name in self.models:
            name = f"{original_name}_{counter}"
            counter += 1
        
        self.models[name] = self.AVAILABLE_MODELS[model_name](**params)
        return name

    def _make_pipeline(self, estimator):
        """Create preprocessing pipeline with SMOTE"""
        sampler = SMOTE(random_state=self.random_state)
        return ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("sampler", sampler),
            ("clf", estimator)
        ])

    def _generate_model_path(self, name):
        """Generate unique model file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{name}_{timestamp}"
        return unique_name

    def _save_model_and_report(self, pipeline, name, report_dict):
        """Save model and report with proper naming"""
        unique_name = self._generate_model_path(name)
        model_path = os.path.join(self.save_dir, f"{unique_name}.{self.save_format}")
        report_path = os.path.join(self.save_dir, f"{unique_name}_report.json")

        # Enhanced report with metadata
        enhanced_report = {
            "model_name": name,
            "unique_id": unique_name,
            "dataset_name": self.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "save_format": self.save_format,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "n_samples": len(self.X),
            "target_column": self.target_column,
            **report_dict
        }

        try:
            # Save model
            if self.save_format == "pkl":
                with open(model_path, "wb") as f:
                    pickle.dump(pipeline, f)
            elif self.save_format == "joblib":
                joblib.dump(pipeline, model_path)
            elif self.save_format == "onnx":
                initial_type = [('float_input', FloatTensorType([None, self.X.shape[1]]))]
                onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
                with open(model_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

            # Save enhanced report as JSON
            with open(report_path, "w") as f:
                json.dump(enhanced_report, f, indent=4)
                
            return unique_name
            
        except Exception as e:
            raise RuntimeError(f"Error saving model: {str(e)}")

    def _cross_val_score_pipeline(self, name, estimator, progress_callback=None):
        """Cross-validation with progress tracking"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        f1_scores = []
        all_preds, all_trues = [], []

        for i, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
            if progress_callback:
                progress = (i / self.n_splits) * 100
                progress_callback(progress)
                
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            pipe = self._make_pipeline(estimator)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average="binary"))
            all_trues.extend(y_test.tolist())
            all_preds.extend(y_pred.tolist())

        # Train final model on full dataset
        final_pipe = self._make_pipeline(estimator)
        final_pipe.fit(self.X, self.y)

        report_dict = {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "cv_scores": [float(score) for score in f1_scores],
            "confusion_matrix": confusion_matrix(all_trues, all_preds).tolist(),
            "classification_report": classification_report(all_trues, all_preds, output_dict=True, digits=5)
        }

        unique_name = self._save_model_and_report(final_pipe, name, report_dict)

        return np.mean(f1_scores), name, final_pipe, report_dict, unique_name

    async def train_all_async(self, progress_callback=None):
        """Asynchronous training of all models"""
        self.results = []
        total_models = len(self.models)
        
        for i, (name, est) in enumerate(self.models.items()):
            try:
                if progress_callback:
                    overall_progress = (i / total_models) * 100
                    await progress_callback(overall_progress, f"Training {name}")
                
                def model_progress(fold_progress):
                    if progress_callback:
                        # Calculate combined progress
                        model_base = (i / total_models) * 100
                        model_specific = (fold_progress / total_models)
                        asyncio.create_task(progress_callback(model_base + model_specific, f"Training {name} - Fold progress: {fold_progress:.1f}%"))
                
                result = self._cross_val_score_pipeline(name, est, model_progress)
                self.results.append(result)
                
                # Small delay to allow other async operations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        self.results.sort(key=lambda x: x[0], reverse=True)
        
        if progress_callback:
            await progress_callback(100, "Training completed")

    def train_all(self):
        """Synchronous training wrapper"""
        return asyncio.run(self.train_all_async())

    def rank_models(self):
        """Return ranked models by performance"""
        return self.results

    def top_model(self):
        """Get the best performing model"""
        if not self.results:
            raise RuntimeError("No models trained yet.")
        return self.results[0]

    def evaluate_top(self):
        """Evaluate and display top model results"""
        top_score, top_name, top_pipe, report, unique_id = self.top_model()
        print(f"\nTop model: {top_name} (ID: {unique_id})")
        print(f"Mean F1 Score: {top_score:.5f}")
        print("Confusion Matrix:\n", np.array(report["confusion_matrix"]))
        print("\nClassification Report:\n", json.dumps(report["classification_report"], indent=2))
        return top_score, top_name, top_pipe, report, unique_id

    def load_model(self, unique_name):
        """Load a saved model by its unique name"""
        model_path = os.path.join(self.save_dir, f"{unique_name}.{self.save_format}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found for '{unique_name}'")
        
        if self.save_format == "pkl":
            with open(model_path, "rb") as f:
                return pickle.load(f)
        elif self.save_format == "joblib":
            return joblib.load(model_path)
        elif self.save_format == "onnx":
            with open(model_path, "rb") as f:
                return f.read()  # Returns ONNX binary

    def list_saved_models(self):
        """List all saved models for this dataset"""
        models = []
        if not os.path.exists(self.save_dir):
            return models
            
        for file in os.listdir(self.save_dir):
            if file.endswith(f".{self.save_format}"):
                model_name = file.replace(f".{self.save_format}", "")
                report_path = os.path.join(self.save_dir, f"{model_name}_report.json")
                
                model_info = {
                    "unique_id": model_name,
                    "file_path": os.path.join(self.save_dir, file),
                    "report_available": os.path.exists(report_path)
                }
                
                if model_info["report_available"]:
                    with open(report_path, "r") as f:
                        report = json.load(f)
                        model_info.update({
                            "model_name": report.get("model_name", "Unknown"),
                            "mean_f1": report.get("mean_f1", 0),
                            "timestamp": report.get("timestamp", "Unknown")
                        })
                
                models.append(model_info)
        
        # Sort by F1 score descending
        return sorted(models, key=lambda x: x.get("mean_f1", 0), reverse=True)

# Task Management for Async Training
class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
    
    def create_task(self, dataset_name: str, model_name: str) -> str:
        """Create a new training task"""
        task_id = str(uuid.uuid4())
        task = TrainingTask(
            task_id=task_id,
            dataset_name=dataset_name,
            model_name=model_name,
            status=TrainingStatus.PENDING,
            progress=0.0
        )
        self.tasks[task_id] = task
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def update_task_progress(self, task_id: str, progress: float, status: TrainingStatus = None):
        """Update task progress"""
        if task_id in self.tasks:
            self.tasks[task_id].progress = progress
            if status:
                self.tasks[task_id].status = status
    
    def complete_task(self, task_id: str, results: Dict):
        """Mark task as completed"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TrainingStatus.COMPLETED
            task.progress = 100.0
            task.end_time = datetime.now()
            task.results = results
    
    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TrainingStatus.FAILED
            task.end_time = datetime.now()
            task.error_message = error_message

# Global task manager instance
task_manager = TaskManager()

# -------------------------
# Example usage and testing
# -------------------------
if __name__ == "__main__":
    # Example with credit card fraud dataset
    manager = MLManager(
        dataset_path="data/sets/creditcard.csv", 
        target_column="Class", 
        drop_columns=["Time"],
        dataset_name="creditcard_fraud"
    )

    # Register multiple models with different configurations
    models_config = [
        ("LogisticRegression_basic", "LogisticRegression", {"max_iter": 2000, "random_state": 1}),
        ("LogisticRegression_l1", "LogisticRegression", {"max_iter": 2000, "penalty": "l1", "solver": "liblinear", "random_state": 1}),
        ("RandomForest_shallow", "RandomForest", {"max_depth": 4, "n_estimators": 100, "random_state": 1}),
        ("RandomForest_deep", "RandomForest", {"max_depth": 8, "n_estimators": 200, "random_state": 1}),
        ("XGBoost_basic", "XGBoost", {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.1, "random_state": 1}),
    ]
    
    for name, model_name, params in models_config:
        try:
            registered_name = manager.register_model(name, model_name, **params)
            print(f"Registered: {registered_name}")
        except Exception as e:
            print(f"Failed to register {name}: {e}")

    # Train all models
    print("\nStarting training...")
    manager.train_all()

    # Show rankings
    print("\n" + "="*50)
    print("MODEL RANKINGS:")
    print("="*50)
    for i, (score, name, _, _, unique_id) in enumerate(manager.rank_models(), 1):
        print(f"{i}. {name} (ID: {unique_id}): {score:.5f}")

    # Evaluate top model
    print("\n" + "="*50)
    print("TOP MODEL EVALUATION:")
    print("="*50)
    manager.evaluate_top()
    
    # List all saved models
    print("\n" + "="*50)
    print("SAVED MODELS:")
    print("="*50)
    saved_models = manager.list_saved_models()
    for model in saved_models:
        print(f"ID: {model['unique_id']}")
        print(f"  Model: {model.get('model_name', 'Unknown')}")
        print(f"  F1 Score: {model.get('mean_f1', 0):.5f}")
        print(f"  Timestamp: {model.get('timestamp', 'Unknown')}")
        print()