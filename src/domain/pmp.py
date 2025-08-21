import os
import pickle
import joblib
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class MLManager:
    AVAILABLE_MODELS = {
        "LogisticRegression": LogisticRegression,
        "KNN": KNeighborsClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "SVC": SVC,
        "XGBoost": XGBClassifier
    }

    SUPPORTED_FORMATS = ["pkl", "joblib", "onnx", "json"]

    def __init__(self, dataset_path, target_column, drop_columns=None, random_state=1, n_splits=5,
                 save_dir="data/models", save_format="pkl"):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.drop_columns = drop_columns or []
        self.random_state = random_state
        self.n_splits = n_splits
        self.save_dir = save_dir
        self.save_format = save_format.lower()
        if self.save_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported save_format: {self.save_format}. Supported: {self.SUPPORTED_FORMATS}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.models = {}
        self.results = []
        self._load_dataset()

    def _load_dataset(self):
        data = pd.read_csv(self.dataset_path)
        if self.drop_columns:
            data = data.drop(columns=[c for c in self.drop_columns if c in data.columns])
        data = data.drop_duplicates()
        self.X = data.drop(columns=[self.target_column]).values
        self.y = data[self.target_column].values

    def register_model(self, name, model_name, **params):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' is not available.")
        self.models[name] = self.AVAILABLE_MODELS[model_name](**params)

    def _make_pipeline(self, estimator):
        sampler = SMOTE(random_state=self.random_state)
        return ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("sampler", sampler),
            ("clf", estimator)
        ])

    def _save_model_and_report(self, pipeline, name, report_dict):
        model_path = os.path.join(self.save_dir, f"{name}.{self.save_format}")
        report_path = os.path.join(self.save_dir, f"{name}_report.json")

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

        # Save report as JSON
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)

    def _cross_val_score_pipeline(self, name, estimator):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        f1_scores = []
        all_preds, all_trues = [], []

        for train_idx, test_idx in tqdm(skf.split(self.X, self.y), total=self.n_splits, desc=f"{name} folds"):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            pipe = self._make_pipeline(estimator)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average="binary"))
            all_trues.extend(y_test.tolist())
            all_preds.extend(y_pred.tolist())

        final_pipe = self._make_pipeline(estimator)
        final_pipe.fit(self.X, self.y)

        report_dict = {
            "mean_f1": float(np.mean(f1_scores)),
            "confusion_matrix": confusion_matrix(all_trues, all_preds).tolist(),
            "classification_report": classification_report(all_trues, all_preds, output_dict=True, digits=5)
        }

        self._save_model_and_report(final_pipe, name, report_dict)

        return np.mean(f1_scores), name, final_pipe, report_dict

    def train_all(self):
        self.results = []
        for name, est in tqdm(self.models.items(), desc="Evaluating models"):
            self.results.append(self._cross_val_score_pipeline(name, est))
        self.results.sort(key=lambda x: x[0], reverse=True)

    def rank_models(self):
        return self.results

    def top_model(self):
        if not self.results:
            raise RuntimeError("No models trained yet.")
        return self.results[0]

    def evaluate_top(self):
        top_score, top_name, top_pipe, report = self.top_model()
        print(f"\nTop model: {top_name}")
        print("Confusion Matrix:\n", np.array(report["confusion_matrix"]))
        print("\nClassification Report:\n", json.dumps(report["classification_report"], indent=4))
        return top_score, top_name, top_pipe, report

    def load_model(self, name):
        model_path = os.path.join(self.save_dir, f"{name}.{self.save_format}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found for '{name}'")
        if self.save_format == "pkl":
            with open(model_path, "rb") as f:
                return pickle.load(f)
        elif self.save_format == "joblib":
            return joblib.load(model_path)
        elif self.save_format == "onnx":
            with open(model_path, "rb") as f:
                return f.read()  # Returns ONNX binary, use onnxruntime to load


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    manager = MLManager(dataset_path="data/sets/creditcard.csv", target_column="Class", drop_columns=["Time"])

    # Register models with optional hyperparameters
    manager.register_model("LogisticRegression", "LogisticRegression", max_iter=2000, random_state=1)
    # manager.register_model("KNN(k=7)", "KNN", n_neighbors=7)
    # manager.register_model("DecisionTree(depth=4)", "DecisionTree", max_depth=4, criterion="entropy", random_state=1)
    # manager.register_model("RandomForest(depth=4)", "RandomForest", max_depth=4, n_estimators=200, random_state=1, n_jobs=-1)
    # manager.register_model("SVC(RBF)", "SVC", kernel="rbf", probability=False, random_state=1)
    # manager.register_model("XGBoost(depth=4)", "XGBoost", max_depth=4, n_estimators=400, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=1, n_jobs=-1)

    # Train all models
    manager.train_all()

    # Show rankings
    print("\nRANKINGS:")
    for score, name, _, _ in manager.rank_models():
        print(f"{name}: {score:.5f}")

    # Evaluate top model
    manager.evaluate_top()
