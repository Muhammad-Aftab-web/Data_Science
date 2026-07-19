import streamlit as st

from authentication import auth
from database import db
from utils.helpers import inject_theme_css, predictions_to_dataframe
from reports.report_generator import dataframe_to_csv_bytes, build_history_pdf

st.set_page_config(page_title="Reports", page_icon="🧾", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("🧾 Reports")

user_id = None if auth.is_admin() else st.session_state["auth_user_id"]
predictions = db.get_predictions(user_id=user_id)
df = predictions_to_dataframe(predictions)

st.write(f"Building a report from **{len(df)}** prediction record(s).")

if df.empty:
    st.info("No predictions available yet.")
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download Full CSV Report",
        data=dataframe_to_csv_bytes(df),
        file_name="full_prediction_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    pdf_bytes = build_history_pdf(st.session_state["auth_full_name"], df)
    st.download_button(
        "⬇️ Download Full PDF Report",
        data=pdf_bytes,
        file_name="full_prediction_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

st.caption(
    "For a single-prediction report with SHAP explanation and safety "
    "recommendations, use the download button at the bottom of the "
    "**Predict** page right after making a prediction."
)
