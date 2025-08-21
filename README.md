# ml

A FastAPI-based service to **upload datasets, register/train models, and perform inference**.  
Supports common ML models, cross-validation, model ranking, and saving models in multiple formats (`pkl`, `joblib`, `onnx`).

---

## Features

- Upload and manage datasets.
- Register multiple ML models per dataset.
- Train models with cross-validation and SMOTE oversampling.
- Rank and evaluate models automatically.
- Serve models for inference (predict unseen data).
- Save models and reports in standard formats.

---

## Supported Models

| Model | Description |
|-------|-------------|
| Decision Tree | Classic tree-based classifier |
| K-Nearest Neighbors (KNN) | Non-parametric, instance-based learning |
| Logistic Regression (LR) | Linear classification with probabilistic output |
| Random Forest | Ensemble of decision trees |
| Support Vector Classifier (SVC) | SVM with customizable kernels |
| XGBoost | Gradient boosting trees |

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

---

## API Endpoints

### Datasets
| Method | Endpoint            | Description            |
| ------ | ------------------- | ---------------------- |
| POST   | `/datasets/upload/` | Upload a CSV dataset   |
| GET    | `/datasets/`        | List uploaded datasets |


### Models
| Method | Endpoint                                    | Description                                 |
| ------ | ------------------------------------------- | ------------------------------------------- |
| GET    | `/models/available`                         | List all available model types              |
| POST   | `/models/register/{dataset_name}`           | Register a model for a dataset              |
| POST   | `/models/train/{dataset_name}/{model_name}` | Train a specific model                      |
| GET    | `/models/list/{dataset_name}`               | List trained models and report availability |


### Inference
| Method | Endpoint                  | Description                            |
| ------ | ------------------------- | -------------------------------------- |
| POST   | `/predict/{dataset_name}` | Predict new data using a trained model |

--- 

## Examples

### Upload Dataset
```bash
curl -X POST "http://127.0.0.1:8000/datasets/upload/" \
-F "file=@data/sets/creditcard.csv"
```

### Register & Train Model
```bash
curl -X POST "http://127.0.0.1:8000/models/register/creditcard" \
-H "Content-Type: application/json" \
-d '{"name": "LR_model", "model_name": "LogisticRegression", "params": {"max_iter":2000}}'

curl -X POST "http://127.0.0.1:8000/models/train/creditcard/LR_model"
```

### Predict
```bash
curl -X POST "http://127.0.0.1:8000/predict/creditcard" \
-H "Content-Type: application/json" \
-d '{"dataset": [[0.1, 0.2, 0.3, ...]], "model_name": "LR_model"}'
```

## Notes
- Models are saved under data/models/{dataset_name}/.
- Reports are saved as {model_name}_report.json.
- Cross-validation uses SMOTE to handle class imbalance.
- Supports pkl, joblib, and onnx formats.

## License
MIT License