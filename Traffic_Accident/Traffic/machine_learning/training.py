"""
machine_learning/training.py
Trains and compares Random Forest, XGBoost, LightGBM, Logistic Regression,
Decision Tree, and MLP Neural Network classifiers. Evaluates each with
Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix, picks the
best model by F1 (macro), and saves it (+ metadata) with Joblib.
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from machine_learning.preprocessing import prepare_training_data, save_encoders

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def get_model_zoo(random_state=42):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300, max_depth=8, random_state=random_state, verbose=-1
        )
    return models


def _safe_roc_auc(y_test, proba, n_classes):
    try:
        if n_classes == 2:
            return roc_auc_score(y_test, proba[:, 1])
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        return roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    n_classes = len(np.unique(y_test))
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": _safe_roc_auc(y_test, proba, n_classes) if proba is not None else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=config.SEVERITY_LEVELS, zero_division=0
        ),
    }
    return metrics


def train_and_compare(df: pd.DataFrame, progress_callback=None):
    """
    Trains every model in the zoo, evaluates it, and returns:
      results: dict[model_name] -> {"model", "metrics", "train_seconds"}
      best_model_name: str
      encoders: dict
    """
    X_train, X_test, y_train, y_test, encoders = prepare_training_data(df)
    models = get_model_zoo()

    results = {}
    total = len(models)
    for i, (name, model) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(i, total, name)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics, "train_seconds": elapsed}

    best_model_name = max(results, key=lambda n: results[n]["metrics"]["f1"])

    save_encoders(encoders)
    return results, best_model_name, encoders, (X_train, X_test, y_train, y_test)


def save_best_model(results: dict, best_model_name: str):
    best_model = results[best_model_name]["model"]
    model_path = os.path.join(config.MODELS_DIR, f"{best_model_name.replace(' ', '_')}.joblib")
    joblib.dump(best_model, model_path)

    meta = {
        "best_model_name": best_model_name,
        "model_path": model_path,
        "metrics": {
            name: {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"}
            for name, r in results.items()
        },
        "all_model_paths": {},
    }
    # Persist every trained model too, so Settings > Change Active Model works
    for name, r in results.items():
        p = os.path.join(config.MODELS_DIR, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(r["model"], p)
        meta["all_model_paths"][name] = p

    joblib.dump(meta, config.BEST_MODEL_META_PATH)
    return model_path, meta


def load_active_model():
    meta = joblib.load(config.BEST_MODEL_META_PATH)
    model = joblib.load(meta["model_path"])
    return model, meta


def set_active_model(model_name: str):
    meta = joblib.load(config.BEST_MODEL_META_PATH)
    if model_name not in meta["all_model_paths"]:
        raise ValueError(f"Model '{model_name}' was not found among trained models.")
    meta["best_model_name"] = model_name
    meta["model_path"] = meta["all_model_paths"][model_name]
    joblib.dump(meta, config.BEST_MODEL_META_PATH)
    return meta
