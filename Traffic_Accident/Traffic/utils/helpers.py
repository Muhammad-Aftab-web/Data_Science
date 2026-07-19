"""
utils/helpers.py
Small reusable helper functions shared across pages (theme CSS, toasts, etc).
"""

import streamlit as st
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def inject_theme_css(dark_mode: bool = False):
    """Lightweight dark/light theme toggle via CSS variable overrides."""
    if dark_mode:
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        div[data-testid="stMetric"] { background-color: #1c1f26; border-radius: 10px; padding: 10px; }
        </style>
        """
    else:
        css = """
        <style>
        div[data-testid="stMetric"] { background-color: #f7f9fc; border-radius: 10px; padding: 10px; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def toast(message: str, icon: str = "✅"):
    try:
        st.toast(message, icon=icon)
    except Exception:
        st.success(message)


def severity_badge(severity: str) -> str:
    color = config.SEVERITY_COLORS.get(severity, "#999")
    return f"<span style='background-color:{color};color:white;padding:3px 10px;border-radius:12px;font-weight:600'>{severity}</span>"


def predictions_to_dataframe(predictions) -> pd.DataFrame:
    rows = []
    for p in predictions:
        rows.append(
            {
                "Prediction ID": p.prediction_id,
                "Date": p.prediction_date,
                "Time": p.prediction_time,
                "City": p.city,
                "Weather": p.weather,
                "Road Condition": p.road_condition,
                "Light Condition": p.light_condition,
                "Traffic Density": p.traffic_density,
                "Speed Limit": p.speed_limit,
                "Road Type": p.road_type,
                "Vehicle Type": p.vehicle_type,
                "Traffic Violation": p.traffic_violation,
                "Seatbelt/Helmet": p.seatbelt_or_helmet,
                "Injuries": p.injuries,
                "Fatalities": p.fatalities,
                "Predicted Severity": p.predicted_severity,
                "Confidence": round(p.confidence_score, 3),
                "Model Used": p.model_used,
                "Created At": p.created_at,
            }
        )
    return pd.DataFrame(rows)


def paginate_dataframe(df: pd.DataFrame, page_size: int = 15, key: str = "page"):
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1, key=key
    )
    start = (page - 1) * page_size
    end = start + page_size
    st.caption(f"Showing rows {start + 1}–{min(end, len(df))} of {len(df)} · Page {page}/{total_pages}")
    return df.iloc[start:end]
