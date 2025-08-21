#!/usr/bin/env python3
"""
Complete ML Workflow Test Script
Demonstrates the full workflow: upload dataset, register models, train, and test inference
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os
from typing import Dict, List, Any

class MLWorkflowTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {}
        
    def log_step(self, step: str, status: str, details: Any = None):
        """Log workflow step results"""
        print(f"\n{'='*50}")
        print(f"STEP: {step}")
        print(f"STATUS: {status}")
        if details:
            print(f"DETAILS: {details}")
        print('='*50)
        
        self.results[step] = {
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
    
    def create_sample_dataset(self, filename: str = "test_creditcard.csv") -> str:
        """Create a sample credit card fraud detection dataset"""
        print("\nCreating sample dataset...")
        
        # Generate synthetic credit card transaction data
        X, y = make_classification(
            n_samples=1100,  # Smaller dataset for faster testing
            n_features=28,  # Similar to real credit card dataset
            n_informative=20,
            n_redundant=5,
            n_clusters_per_class=1,
            n_classes=2,
            class_sep=0.8,
            flip_y=0.01,  # Small amount of noise
            random_state=42
        )
        
        # Create feature names similar to credit card dataset
        feature_names = [f"V{i+1}" for i in range(28)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add Amount column (simulating transaction amounts)
        df['Amount'] = np.random.lognormal(3, 1.5, size=len(df))
        df['Amount'] = np.round(df['Amount'], 2)
        
        # Add Class column (0: normal, 1: fraud)
        df['Class'] = y
        
        # Make it imbalanced like real fraud detection
        fraud_indices = df[df['Class'] == 1].index
        normal_indices = df[df['Class'] == 0].index
        
        # Keep only small percentage as fraud
        fraud_keep = np.random.choice(fraud_indices, size=int(len(fraud_indices) * 0.1), replace=False)
        indices_to_keep = np.concatenate([normal_indices, fraud_keep])
        df = df.loc[indices_to_keep].reset_index(drop=True)
        
        # Save dataset
        dataset_path = f"temp_{filename}"
        df.to_csv(dataset_path, index=False)
        
        self.log_step(
            "dataset_creation", 
            "success", 
            f"Created {len(df)} samples with {len(feature_names)+2} features. Fraud rate: {df['Class'].mean():.2%}"
        )
        
        return dataset_path
    
    def upload_dataset(self, file_path: str) -> Dict[str, Any]:
        """Upload dataset to the API"""
        print("\nUploading dataset...")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'text/csv')}
                response = self.session.post(f"{self.base_url}/datasets/upload/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                self.log_step("dataset_upload", "success", result)
                return result
            else:
                self.log_step("dataset_upload", "failed", {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return {}
                
        except Exception as e:
            self.log_step("dataset_upload", "failed", str(e))
            return {}
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset information"""
        print("\nGetting dataset information...")
        
        try:
            response = self.session.get(f"{self.base_url}/datasets/{dataset_name}")
            
            if response.status_code == 200:
                result = response.json()
                self.log_step("dataset_info", "success", {
                    "columns": len(result["columns"]),
                    "rows": result["rows"],
                    "target_suggestions": result["target_suggestions"]
                })
                return result
            else:
                self.log_step("dataset_info", "failed", {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return {}
                
        except Exception as e:
            self.log_step("dataset_info", "failed", str(e))
            return {}
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available ML models"""
        print("\nGetting available models...")
        
        try:
            response = self.session.get(f"{self.base_url}/models/available")
            
            if response.status_code == 200:
                result = response.json()
                self.log_step("available_models", "success", f"Found {len(result['models'])} model types")
                return result
            else:
                self.log_step("available_models", "failed", {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return {}
                
        except Exception as e:
            self.log_step("available_models", "failed", str(e))
            return {}
    
    def register_models(self, dataset_name: str) -> List[str]:
        """Register multiple models for training"""
        print("\nRegistering models...")
        
        model_configs = [
            {
                "name": "LogisticRegression_L1",
                "model_name": "LogisticRegression",
                "params": {"penalty": "l1", "solver": "liblinear", "max_iter": 2000, "random_state": 42},
                "target_column": "Class",
                "drop_columns": []
            },
            {
                "name": "LogisticRegression_L2", 
                "model_name": "LogisticRegression",
                "params": {"penalty": "l2", "max_iter": 2000, "random_state": 42},
                "target_column": "Class",
                "drop_columns": []
            },
            {
                "name": "RandomForest_Shallow",
                "model_name": "RandomForest", 
                "params": {"n_estimators": 50, "max_depth": 4, "random_state": 42, "n_jobs": -1},
                "target_column": "Class",
                "drop_columns": []
            },
            {
                "name": "RandomForest_Deep",
                "model_name": "RandomForest",
                "params": {"n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1},
                "target_column": "Class", 
                "drop_columns": []
            },
            {
                "name": "XGBoost_Conservative",
                "model_name": "XGBoost",
                "params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
                "target_column": "Class",
                "drop_columns": []
            }
        ]
        
        registered_model_ids = []
        
        for config in model_configs:
            try:
                response = self.session.post(
                    f"{self.base_url}/models/register/{dataset_name}",
                    json=config
                )
                
                if response.status_code == 200:
                    result = response.json()
                    registered_model_ids.append(result["model_id"])
                    print(f"Registered: {config['name']} -> {result['model_id']}")
                else:
                    print(f"Failed to register {config['name']}: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Error registering {config['name']}: {str(e)}")
        
        self.log_step("model_registration", "success", f"Registered {len(registered_model_ids)} models")
        return registered_model_ids
    
    def train_models(self, dataset_name: str, model_ids: List[str]) -> str:
        """Start model training - FIXED VERSION"""
        print("\nStarting model training...")
        
        # Create the training configuration
        training_config = {
            "target_column": "Class",
            "drop_columns": [],
            "n_splits": 5,
            "random_state": 42,
            "save_format": "pkl"
        }
        
        try:
            # FIXED: Pass model_ids in the request body, not as query params
            request_data = {
                **training_config,  # Spread the training config into the request body
                "model_ids": model_ids  # Include model_ids in the body
            }
            
            response = self.session.post(
                f"{self.base_url}/models/train/{dataset_name}",
                json=request_data
            )
            
            print(f"Training request - Status: {response.status_code}")
            print(f"Request data: {json.dumps(request_data, indent=2)}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]
                self.log_step("training_started", "success", {
                    "task_id": task_id,
                    "poll_url": result["poll_url"],
                    "message": result["message"]
                })
                return task_id
            else:
                self.log_step("training_started", "failed", {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return ""
                
        except Exception as e:
            self.log_step("training_started", "failed", str(e))
            return ""
    
    def poll_training_status(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Poll training status until completion"""
        print(f"\nPolling training status for task {task_id}...")
        
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.base_url}/models/tasks/{task_id}")
                
                if response.status_code == 200:
                    status = response.json()
                    
                    # Show progress if changed
                    if status["progress"] != last_progress:
                        print(f"Progress: {status['progress']:.1f}% - Status: {status['status']}")
                        last_progress = status["progress"]
                    
                    if status["status"] == "completed":
                        self.log_step("training_completed", "success", {
                            "duration": time.time() - start_time,
                            "results": status["results"]
                        })
                        return status
                    elif status["status"] == "failed":
                        self.log_step("training_completed", "failed", status["error_message"])
                        return status
                    
                    # Wait before next poll
                    time.sleep(2)
                else:
                    print(f"Error polling status: {response.status_code}")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Error polling: {str(e)}")
                time.sleep(5)
        
        self.log_step("training_completed", "timeout", f"Training timed out after {timeout} seconds")
        return {}
    
    def list_trained_models(self, dataset_name: str) -> Dict[str, Any]:
        """List all trained models for the dataset"""
        print("\nListing trained models...")
        
        try:
            response = self.session.get(f"{self.base_url}/models/list/{dataset_name}")
            
            if response.status_code == 200:
                result = response.json()
                trained_count = len(result.get("trained_models", []))
                self.log_step("list_models", "success", f"Found {trained_count} trained models")
                return result
            else:
                self.log_step("list_models", "failed", {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return {}
                
        except Exception as e:
            self.log_step("list_models", "failed", str(e))
            return {}
    
    def test_inference(self, dataset_name: str, model_id: str, test_data: List[List[float]]) -> Dict[str, Any]:
        """Test model inference"""
        print(f"\nTesting inference with model {model_id}...")
        
        inference_request = {
            "instances": test_data,
            "model_id": model_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/inference/{dataset_name}",
                json=inference_request
            )
            
            if response.status_code == 200:
                result = response.json()
                self.log_step("inference", "success", {
                    "predictions": result["predictions"],
                    "probabilities": result.get("prediction_probabilities"),
                    "feature_count": len(result.get("feature_names", []))
                })
                return result
            else:
                self.log_step("inference", "failed", {
                    "status_code": response.status_code,
                    "response": response.text
                })
                return {}
                
        except Exception as e:
            self.log_step("inference", "failed", str(e))
            return {}
    
    def run_complete_workflow(self) -> Dict[str, Any]:
        """Run the complete ML workflow"""
        print("\nStarting Complete ML Workflow Test")
        print("=" * 60)
        
        # Step 1: Create sample dataset
        dataset_file = self.create_sample_dataset()
        
        # Step 2: Upload dataset
        upload_result = self.upload_dataset(dataset_file)
        if not upload_result:
            print("Dataset upload failed. Stopping workflow.")
            return self.results
        
        dataset_name = upload_result["filename"]
        print(f"Using dataset: {dataset_name}")
        
        # Step 3: Get dataset info
        dataset_info = self.get_dataset_info(dataset_name)
        if not dataset_info:
            print("Could not get dataset info. Stopping workflow.")
            return self.results
        
        # Step 4: Get available models
        available_models = self.get_available_models()
        if not available_models:
            print("Could not get available models. Stopping workflow.")
            return self.results
        
        # Step 5: Register models
        model_ids = self.register_models(dataset_name)
        if not model_ids:
            print("No models registered. Stopping workflow.")
            return self.results
        
        print(f"Registered {len(model_ids)} models: {model_ids}")
        
        # Step 6: Start training
        task_id = self.train_models(dataset_name, model_ids)
        if not task_id:
            print("Training failed to start. Stopping workflow.")
            return self.results
        
        print(f"Training started with task ID: {task_id}")
        
        # Step 7: Poll training status
        training_result = self.poll_training_status(task_id)
        if not training_result or training_result.get("status") != "completed":
            print("Training did not complete successfully.")
            return self.results
        
        # Step 8: List trained models
        models_list = self.list_trained_models(dataset_name)
        if not models_list:
            print("Could not list trained models.")
            return self.results
        
        # Step 9: Test inference with best model
        trained_models = models_list.get("trained_models", [])
        if trained_models:
            best_model = trained_models[0]  # First model is best (sorted by F1 score)
            print(f"Testing inference with best model: {best_model.get('name', 'Unknown')}")
            
            # Create test data (using feature structure from dataset)
            n_features = len(dataset_info["columns"]) - 1  # Exclude target column
            test_data = np.random.randn(3, n_features).tolist()  # 3 test samples
            
            inference_result = self.test_inference(
                dataset_name, 
                best_model["unique_id"], 
                test_data
            )
        
        # Cleanup temp file
        try:
            os.remove(dataset_file)
            print(f"🧹 Cleaned up temporary file: {dataset_file}")
        except:
            pass
        
        # Final summary
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print workflow summary"""
        print("\n" + "="*60)
        print("WORKFLOW SUMMARY")
        print("="*60)
        
        total_steps = len(self.results)
        successful_steps = sum(1 for r in self.results.values() if r["status"] == "success")
        
        print(f"Total Steps: {total_steps}")
        print(f"Successful Steps: {successful_steps}")
        print(f"Success Rate: {successful_steps/total_steps*100:.1f}%")
        
        print("\nStep Details:")
        for step, result in self.results.items():
            status_enum = "✅" if result["status"] == "success" else "❌" if result["status"] == "failed" else "⏸️"
            print(f"{status_enum} {step}: {result['status']}")
            
        # Training results if available
        if "training_completed" in self.results and self.results["training_completed"]["status"] == "success":
            training_details = self.results["training_completed"]["details"]
            if training_details and "results" in training_details:
                rankings = training_details["results"].get("rankings", [])
                if rankings:
                    print(f"\nModel Performance Rankings:")
                    for i, model in enumerate(rankings[:3], 1):  # Top 3
                        print(f"{i}. {model['name']}: F1={model['f1_score']:.4f} (±{model['std_f1']:.4f})")
        
        print("\n" + "="*60)

def main():
    """Main function to run the workflow test"""
    print("ML Workflow Tester - Fixed Version")
    print("="*50)
    
    tester = MLWorkflowTester()
    
    # Check API health
    try:
        response = requests.get(f"{tester.base_url}/health")
        if response.status_code != 200:
            print("API is not running. Please start the FastAPI server first.")
            print("Run: uvicorn src.app:app --reload")
            return
        else:
            print("API is running and healthy!")
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Please start the FastAPI server first.")
        print("Run: uvicorn src.app:app --reload")
        return
    
    # Run complete workflow
    print("Starting workflow test...")
    results = tester.run_complete_workflow()
    
    # Save results to file
    with open("workflow_test_results.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"\nResults saved to workflow_test_results.json")

if __name__ == "__main__":
    main()