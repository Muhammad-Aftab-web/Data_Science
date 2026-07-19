import os
import streamlit as st
import pandas as pd
import plotly.express as px

import config
from authentication import auth
from utils.helpers import inject_theme_css, toast
from machine_learning.preprocessing import load_dataset
from machine_learning.training import train_and_compare, save_best_model

st.set_page_config(page_title="Train Models", page_icon="🤖", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("🤖 Machine Learning: Train & Compare Models")

df = st.session_state.get("processed_dataset")
if df is None and os.path.exists(config.PROCESSED_DATASET_PATH):
    df = load_dataset(config.PROCESSED_DATASET_PATH)
    st.session_state["processed_dataset"] = df

if df is None:
    st.warning("No processed dataset found. Go to **Dataset Management** first.")
    st.stop()

st.caption(f"Training on {len(df)} rows.")

if st.button("🚀 Train & Compare All Models", type="primary"):
    progress_bar = st.progress(0.0, text="Starting...")

    def _cb(i, total, name):
        progress_bar.progress(i / total, text=f"Training {name} ({i}/{total})...")

    with st.spinner("Training in progress — this can take a minute..."):
        results, best_name, encoders, splits = train_and_compare(df, progress_callback=_cb)
        model_path, meta = save_best_model(results, best_name)

    progress_bar.progress(1.0, text="Done!")
    st.session_state["last_training_results"] = {
        name: r["metrics"] for name, r in results.items()
    }
    st.session_state["best_model_name"] = best_name
    toast(f"Best model: {best_name} 🎉")

results = st.session_state.get("last_training_results")
best_name = st.session_state.get("best_model_name")

if results:
    st.subheader("🏆 Best Model")
    st.success(f"**{best_name}** was selected as the best performer (highest macro F1 score).")

    st.subheader("📋 Model Comparison")
    comp_rows = []
    for name, m in results.items():
        comp_rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1 Score": round(m["f1"], 4),
                "ROC-AUC": round(m["roc_auc"], 4) if m["roc_auc"] == m["roc_auc"] else "N/A",
            }
        )
    comp_df = pd.DataFrame(comp_rows).sort_values("F1 Score", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        comp_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"]),
        x="Model", y="value", color="variable", barmode="group",
        labels={"value": "Score", "variable": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Detailed Report per Model")
    selected = st.selectbox("Choose a model", list(results.keys()), index=list(results.keys()).index(best_name))
    m = results[selected]
    c1, c2 = st.columns(2)
    with c1:
        st.text("Classification Report")
        st.code(m["classification_report"])
    with c2:
        st.text("Confusion Matrix")
        cm = m["confusion_matrix"]
        cm_fig = px.imshow(
            cm, text_auto=True, x=config.SEVERITY_LEVELS, y=config.SEVERITY_LEVELS,
            labels=dict(x="Predicted", y="Actual", color="Count"), color_continuous_scale="Blues",
        )
        st.plotly_chart(cm_fig, use_container_width=True)
else:
    st.info("Click **Train & Compare All Models** to begin.")

st.divider()