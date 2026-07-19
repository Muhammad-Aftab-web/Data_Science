"""
machine_learning/predict.py
Loads the active saved model + encoders, runs a single prediction, computes
a SHAP explanation, and produces plain-language safety recommendations.
"""

import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from machine_learning.preprocessing import load_encoders, encode_single_record
from machine_learning.training import load_active_model

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


PART_OF_DAY_BY_HOUR = [
    (5, 11, "Morning"),
    (12, 16, "Afternoon"),
    (17, 20, "Evening"),
    (21, 23, "Night"),
    (0, 4, "Night"),
]


def derive_time_features(date: dt.date, time_: dt.time):
    hour = time_.hour
    day_of_week = date.strftime("%A")
    month = date.strftime("%B")
    part_of_day = next(
        (label for lo, hi, label in PART_OF_DAY_BY_HOUR if lo <= hour <= hi), "Night"
    )
    return {"Hour": hour, "Part_of_Day": part_of_day, "Day_of_Week": day_of_week, "Month": month}


def build_feature_record(form_data: dict, date: dt.date, time_: dt.time) -> dict:
    derived = derive_time_features(date, time_)
    record = {
        "Weather": form_data["weather"],
        "Road_Condition": form_data["road_condition"],
        "Light_Condition": form_data["light_condition"],
        "Traffic_Density": form_data["traffic_density"],
        "Speed_Limit": form_data["speed_limit"],
        "Road_Type": form_data["road_type"],
        "Vehicle_Type": form_data["vehicle_type"],
        "Traffic_Violation": form_data["traffic_violation"],
        "Seatbelt_or_Helmet": form_data["seatbelt_or_helmet"],
        "Injuries": form_data["injuries"],
        "Fatalities": form_data["fatalities"],
        **derived,
    }
    return record


def predict_severity(form_data: dict, date: dt.date, time_: dt.time):
    """
    Returns a dict with: severity, probabilities, confidence, model_used,
    shap_values (top contributing features), feature_record.
    """
    model, meta = load_active_model()
    encoders = load_encoders()

    record = build_feature_record(form_data, date, time_)
    X = encode_single_record(record, encoders)

    pred_encoded = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    target_encoder = encoders[config.TARGET_COLUMN]
    severity = target_encoder.inverse_transform([pred_encoded])[0]

    probabilities = {}
    if proba is not None:
        classes = target_encoder.inverse_transform(np.arange(len(proba)))
        probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    confidence = max(probabilities.values()) if probabilities else 1.0

    shap_top_features = compute_shap_top_features(model, X)

    return {
        "severity": severity,
        "probabilities": probabilities,
        "confidence": confidence,
        "model_used": meta["best_model_name"],
        "shap_top_features": shap_top_features,
        "feature_record": record,
        "X_encoded": X,
    }


def compute_shap_top_features(model, X: pd.DataFrame, top_n: int = 5):
    """Best-effort SHAP explanation. Falls back to model feature_importances_
    if SHAP or the specific explainer isn't available for this model type."""
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                # multi-class: average |shap| across classes for the predicted row
                arr = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
            else:
                arr = np.abs(shap_values[0])
            pairs = sorted(zip(config.FEATURE_COLUMNS, arr), key=lambda p: p[1], reverse=True)
            return [{"feature": f, "impact": float(v)} for f, v in pairs[:top_n]]
        except Exception:
            pass

    # Fallback: global feature importances (tree-based models only)
    if hasattr(model, "feature_importances_"):
        pairs = sorted(
            zip(config.FEATURE_COLUMNS, model.feature_importances_),
            key=lambda p: p[1],
            reverse=True,
        )
        return [{"feature": f, "impact": float(v)} for f, v in pairs[:top_n]]

    return []


def safety_recommendations(record: dict, severity: str) -> list[str]:
    """Simple rule-based recommendations layered on top of the ML prediction."""
    tips = []
    if record.get("Seatbelt_or_Helmet") == "No":
        tips.append("Always wear a seatbelt or helmet — it is one of the single biggest predictors of reduced injury severity.")
    if record.get("Traffic_Violation") not in (None, "None"):
        tips.append(f"Avoid '{record['Traffic_Violation']}' — traffic violations sharply increase severity risk.")
    if record.get("Speed_Limit", 0) and record.get("Speed_Limit") >= 80:
        tips.append("Reduce speed in this zone; higher speed limits correlate with more severe outcomes.")
    if record.get("Weather") in ("Rain", "Fog", "Dust Storm", "Snow"):
        tips.append(f"Exercise extra caution in {record['Weather'].lower()} conditions — reduce speed and increase following distance.")
    if record.get("Road_Condition") in ("Wet", "Icy", "Damaged", "Under Construction"):
        tips.append(f"Road condition is '{record['Road_Condition']}' — maintain a safe following distance and avoid sudden braking.")
    if record.get("Light_Condition") in ("Dark - Unlit", "Dark - Lit", "Dusk/Dawn"):
        tips.append("Use headlights and stay extra alert during low-light conditions.")
    if severity == "High":
        tips.append("This scenario is classified as High severity — prioritize preventive measures and consider alternate routes/timing.")
    if not tips:
        tips.append("Conditions look relatively favorable — continue following standard road safety practices.")
    return tips
