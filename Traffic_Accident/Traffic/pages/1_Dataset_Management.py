import streamlit as st
import pandas as pd

import config
from authentication import auth
from utils.helpers import inject_theme_css, toast
from machine_learning.preprocessing import (
    load_dataset,
    basic_dataset_stats,
    handle_missing_values,
)

st.set_page_config(page_title="Dataset Management", page_icon="🗂️", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("🗂️ Dataset Management")

st.markdown(
    f"""
The training dataset should contain these columns. Only the **feature** columns are
used for model training — the **context** columns are kept for storage, mapping,
and reporting only.

- **Prediction features:** {", ".join(config.FEATURE_COLUMNS)}
- **Target:** {config.TARGET_COLUMN}
- **Context-only (never used as features):** {", ".join(config.CONTEXT_COLUMNS)}
"""
)

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded is not None:
    df = load_dataset(uploaded)
    st.session_state["raw_dataset"] = df
    df.to_csv(config.RAW_DATASET_PATH, index=False)
    toast("Dataset uploaded and saved.")

df = st.session_state.get("raw_dataset")
if df is None and st.button("Load previously uploaded dataset (if any)"):
    try:
        df = load_dataset(config.RAW_DATASET_PATH)
        st.session_state["raw_dataset"] = df
    except FileNotFoundError:
        st.warning("No previously saved dataset found.")

if df is not None:
    st.subheader("👀 Preview")
    st.dataframe(df.head(20), use_container_width=True)

    missing_cols = [c for c in config.FEATURE_COLUMNS + [config.TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        st.error(f"Dataset is missing required columns: {', '.join(missing_cols)}")
    else:
        st.success("All required columns are present. ✅")

        stats = basic_dataset_stats(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", stats["rows"])
        c2.metric("Columns", stats["columns"])
        c3.metric("Missing Values", stats["missing_values"])

        st.subheader("📊 Statistics")
        st.write("**Severity distribution:**", stats["severity_distribution"])
        with st.expander("Missing values by column"):
            st.write({k: v for k, v in stats["missing_by_column"].items() if v > 0} or "None 🎉")
        with st.expander("Column data types"):
            st.write(stats["dtypes"])

        st.subheader("🧹 Preprocessing")
        if st.button("Handle Missing Values & Save Processed Dataset", type="primary"):
            cleaned = handle_missing_values(df)
            cleaned.to_csv(config.PROCESSED_DATASET_PATH, index=False)
            st.session_state["processed_dataset"] = cleaned
            toast(f"Processed dataset saved ({len(cleaned)} rows).")
            st.dataframe(cleaned.head(10), use_container_width=True)

        st.caption(
            "Once a processed dataset is saved, head to **Train Models** to run the "
            "full model comparison pipeline."
        )
else:
    st.info("Upload a CSV to get started.")
