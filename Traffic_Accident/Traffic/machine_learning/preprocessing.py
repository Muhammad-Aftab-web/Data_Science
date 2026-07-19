"""
machine_learning/preprocessing.py
Handles missing-value cleanup, categorical encoding, and train/test split.
Only FEATURE_COLUMNS are ever used to train the model; CONTEXT_COLUMNS
(Date, City, Latitude, Longitude) are preserved separately for storage /
visualization but are excluded from the model matrix, per project spec.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_dataset(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    return df


def basic_dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "severity_distribution": (
            df[config.TARGET_COLUMN].value_counts().to_dict()
            if config.TARGET_COLUMN in df.columns
            else {}
        ),
    }


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    for col in config.NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    # Drop rows still missing the target
    if config.TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[config.TARGET_COLUMN])
    return df


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict | None = None):
    """
    Label-encode categorical FEATURE_COLUMNS (and the target, if present).
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders = encoders or {}

    for col in config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    if config.TARGET_COLUMN in df.columns:
        if fit:
            le_target = LabelEncoder()
            le_target.fit(config.SEVERITY_LEVELS)  # fixed, deterministic order
            df[config.TARGET_COLUMN] = le_target.transform(df[config.TARGET_COLUMN].astype(str))
            encoders[config.TARGET_COLUMN] = le_target
        else:
            le_target = encoders[config.TARGET_COLUMN]
            df[config.TARGET_COLUMN] = le_target.transform(df[config.TARGET_COLUMN].astype(str))

    return df, encoders


def save_encoders(encoders: dict, path: str = None):
    path = path or config.ENCODERS_PATH
    joblib.dump(encoders, path)


def load_encoders(path: str = None) -> dict:
    path = path or config.ENCODERS_PATH
    return joblib.load(path)


def prepare_training_data(df: pd.DataFrame):
    """
    Full pipeline: clean -> encode -> split into X_train, X_test, y_train, y_test.
    Also returns the fitted encoders and the feature column order.
    """
    df = handle_missing_values(df)
    encoded_df, encoders = encode_features(df, fit=True)

    X = encoded_df[config.FEATURE_COLUMNS]
    y = encoded_df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, encoders


def encode_single_record(record: dict, encoders: dict) -> pd.DataFrame:
    """Encode a single prediction-form record (dict) into a one-row DataFrame
    with columns matching FEATURE_COLUMNS, ready for model.predict()."""
    df = pd.DataFrame([record])
    encoded_df, _ = encode_features(df, fit=False, encoders=encoders)
    return encoded_df[config.FEATURE_COLUMNS]
