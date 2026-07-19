import streamlit as st

import config
from authentication import auth
from database import db
from utils.helpers import inject_theme_css, toast, predictions_to_dataframe, paginate_dataframe
from reports.report_generator import dataframe_to_csv_bytes, build_history_pdf

st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("📜 Prediction History")

is_admin = auth.is_admin()
user_id = None if is_admin else st.session_state["auth_user_id"]

st.subheader("🔍 Filters")
c1, c2, c3, c4 = st.columns(4)
city_filter = c1.selectbox("City", ["All"] + sorted(config.CITY_COORDS.keys()))
severity_filter = c2.selectbox("Severity", ["All"] + config.SEVERITY_LEVELS)
date_from = c3.date_input("From Date", value=None)
date_to = c4.date_input("To Date", value=None)

filters = {}
if city_filter != "All":
    filters["city"] = city_filter
if severity_filter != "All":
    filters["severity"] = severity_filter
if date_from:
    filters["date_from"] = date_from
if date_to:
    filters["date_to"] = date_to

predictions = db.get_predictions(user_id=user_id, filters=filters)
df = predictions_to_dataframe(predictions)

search = st.text_input("🔎 Search (city, weather, vehicle type, model...)")
if search and not df.empty:
    mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False)).any(axis=1)
    df = df[mask]

st.caption(f"{len(df)} record(s) found." + (" (viewing ALL users — Administrator)" if is_admin else ""))

if not df.empty:
    page_df = paginate_dataframe(df, page_size=15)
    st.dataframe(page_df, use_container_width=True, hide_index=True)

    st.subheader("📤 Export")
    ce1, ce2, ce3 = st.columns(3)
    with ce1:
        st.download_button(
            "⬇️ Export CSV", data=dataframe_to_csv_bytes(df),
            file_name="prediction_history.csv", mime="text/csv", use_container_width=True,
        )
    with ce2:
        pdf_bytes = build_history_pdf(st.session_state["auth_full_name"], df)
        st.download_button(
            "⬇️ Export PDF", data=pdf_bytes,
            file_name="prediction_history.pdf", mime="application/pdf", use_container_width=True,
        )
    with ce3:
        del_id = st.number_input("Prediction ID to delete", min_value=0, value=0, step=1)
        if st.button("🗑️ Delete", use_container_width=True) and del_id:
            ok = db.delete_prediction(del_id, user_id=None if is_admin else st.session_state["auth_user_id"])
            if ok:
                toast("Prediction deleted.")
                st.rerun()
            else:
                st.error("Prediction not found or you don't have permission to delete it.")
else:
    st.info("No predictions match your filters yet.")
