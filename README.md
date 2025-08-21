# ml

A FastAPI-based service to **upload datasets, register/train models, and perform inference**.  
Supports common ML models, cross-validation, model ranking, and saving models in multiple formats (`pkl`, `joblib`, `onnx`).

---

## Features

- **Dataset Management**: Upload, list, analyze, and delete CSV datasets
- **Model Registration**: Register multiple model configurations per dataset
- **Asynchronous Training**: Train models with real-time progress tracking
- **Automatic Evaluation**: Cross-validation with SMOTE oversampling for imbalanced datasets
- **Model Ranking**: Automatic ranking by F1-score performance
- **Multiple Formats**: Save models in PKL, Joblib, or ONNX format
- **Inference API**: Make predictions on new data using trained models
- **Comprehensive Reports**: Detailed performance metrics and confusion matrices

---

## Supported Models

| Model | Library | Description |
|-------|---------|-------------|
| LogisticRegression | scikit-learn | Linear classifier with regularization options |
| Ridge | scikit-learn | Ridge regression for continuous targets |
| Lasso | scikit-learn | Lasso regression with L1 regularization |
| KNN | scikit-learn | K-Nearest Neighbors classifier |
| DecisionTree | scikit-learn | Single decision tree classifier |
| RandomForest | scikit-learn | Ensemble of decision trees |
| GradientBoosting | scikit-learn | Gradient boosting classifier |
| AdaBoost | scikit-learn | Adaptive boosting classifier |
| SVC | scikit-learn | Support Vector Classifier |
| GaussianNB | scikit-learn | Gaussian Naive Bayes |
| XGBoost | xgboost | Extreme gradient boosting |
| LightGBM | lightgbm | Microsoft's gradient boosting (optional) |
| CatBoost | catboost | Yandex's gradient boosting (optional) |

---

## Prepare environment
```bash
pip install -r requirements.txt
```

## CLI
Train and evaluate models from the command line:
```bash
python src/domain/model_manager.py
```

## API
Run locally with Uvicorn:
```bash
uvicorn src.app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

## Docker
Build and run the service with Docker Compose:
```bash
docker compose up --build
```

## Testing

Run the complete workflow test:
```bash
python tests/e2e.py
```

---

## API Endpoints

### Dataset Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/datasets/upload/` | Upload a CSV dataset |
| GET | `/datasets/` | List all uploaded datasets |
| GET | `/datasets/{dataset_name}` | Get dataset information and column analysis |
| DELETE | `/datasets/{dataset_name}` | Delete a dataset |

### Model Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models/available` | List available model types and parameters |
| POST | `/models/register/{dataset_name}` | Register a model configuration |
| POST | `/models/train/{dataset_name}` | Start asynchronous model training |
| GET | `/models/tasks/{task_id}` | Check training task status and progress |
| GET | `/models/list/{dataset_name}` | List trained models with performance metrics |
| DELETE | `/models/trained/{dataset_name}/{model_id}` | Delete a trained model |

### Inference
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/inference/{dataset_name}` | Make predictions using trained models |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check endpoint |
| GET | `/` | API information |

--- 

## Examples

### 1. Upload Dataset
```bash
curl -X POST "http://localhost:8000/datasets/upload/" \
  -F "file=@path/to/your/dataset.csv"
```

### 2. Register Models
```bash
# Register Logistic Regression with L1 penalty
curl -X POST "http://localhost:8000/models/register/your_dataset.csv" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LogReg_L1",
    "model_name": "LogisticRegression",
    "params": {
      "penalty": "l1",
      "solver": "liblinear",
      "max_iter": 2000,
      "random_state": 42
    },
    "target_column": "target",
    "drop_columns": ["id", "timestamp"]
  }'

# Register Random Forest
curl -X POST "http://localhost:8000/models/register/your_dataset.csv" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "RandomForest_Deep",
    "model_name": "RandomForest",
    "params": {
      "n_estimators": 100,
      "max_depth": 8,
      "random_state": 42
    },
    "target_column": "target",
    "drop_columns": []
  }'
```

### 3. Train Models
```bash
curl -X POST "http://localhost:8000/models/train/your_dataset.csv" \
  -H "Content-Type: application/json" \
  -d '{
    "target_column": "target",
    "drop_columns": [],
    "n_splits": 5,
    "random_state": 42,
    "save_format": "pkl",
    "model_ids": ["LogReg_L1_20240101_120000", "RandomForest_Deep_20240101_120001"]
  }'
```

### 4. Check Training Progress
```bash
curl -X GET "http://localhost:8000/models/tasks/{task_id}"
```

### 5. Make Predictions
```bash
curl -X POST "http://localhost:8000/inference/your_dataset.csv" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      [1.2, 0.5, -0.3, 2.1, 0.8],
      [0.1, 1.5, 0.7, -1.2, 0.3]
    ],
    "model_id": "LogReg_L1_20240101_120000"
  }'
```

## Python Client Example

```python
import requests
import pandas as pd

class MLAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def upload_dataset(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(f"{self.base_url}/datasets/upload/", files=files)
        return response.json()
    
    def register_model(self, dataset_name, model_config):
        response = self.session.post(
            f"{self.base_url}/models/register/{dataset_name}",
            json=model_config
        )
        return response.json()
    
    def train_models(self, dataset_name, training_config):
        response = self.session.post(
            f"{self.base_url}/models/train/{dataset_name}",
            json=training_config
        )
        return response.json()
    
    def get_task_status(self, task_id):
        response = self.session.get(f"{self.base_url}/models/tasks/{task_id}")
        return response.json()
    
    def predict(self, dataset_name, instances, model_id):
        response = self.session.post(
            f"{self.base_url}/inference/{dataset_name}",
            json={"instances": instances, "model_id": model_id}
        )
        return response.json()

# Example usage
client = MLAPIClient()

# Upload dataset
upload_result = client.upload_dataset("data.csv")
dataset_name = upload_result["filename"]

# Register model
model_config = {
    "name": "MyLogReg",
    "model_name": "LogisticRegression",
    "params": {"max_iter": 2000},
    "target_column": "target",
    "drop_columns": []
}
registration = client.register_model(dataset_name, model_config)

# Train model
training_config = {
    "target_column": "target",
    "drop_columns": [],
    "n_splits": 5,
    "random_state": 42,
    "save_format": "pkl",
    "model_ids": [registration["model_id"]]
}
training = client.train_models(dataset_name, training_config)

# Check status and make predictions when ready
# ... (poll task status until completed)
# predictions = client.predict(dataset_name, [[1,2,3,4,5]], model_id)
```

## Key Features

### Preprocessing Pipeline
- **StandardScaler**: Automatic feature scaling
- **SMOTE**: Synthetic oversampling for imbalanced datasets
- **Missing Value Handling**: Automatic dropna for data cleaning

### Cross-Validation
- **Stratified K-Fold**: Maintains class distribution across folds
- **F1-Score Metric**: Primary evaluation metric for binary classification
- **Comprehensive Reports**: Confusion matrices and classification reports

### Asynchronous Training
- **Background Tasks**: Non-blocking model training
- **Progress Tracking**: Real-time training progress updates
- **Task Management**: Unique task IDs with status monitoring

### Model Persistence
- **Multiple Formats**: PKL, Joblib, ONNX support
- **Unique Naming**: Timestamp-based unique model IDs
- **Metadata Storage**: Complete training reports with model info

## Notes
- Models are saved under data/models/{dataset_name}/.
- Reports are saved as {model_name}_report.json.
- Cross-validation uses SMOTE to handle class imbalance.
- Supports pkl, joblib, and onnx formats.
- Models saved to onnx may be used in static sites with wasm.

## License
MIT License - see LICENSE file for details