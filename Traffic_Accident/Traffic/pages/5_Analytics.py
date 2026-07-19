import pandas as pd
import streamlit as st
import plotly.express as px

import config
from authentication import auth
from database import db
from utils.helpers import inject_theme_css, predictions_to_dataframe

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("📈 Analytics Dashboard")

user_id = None if auth.is_admin() else st.session_state["auth_user_id"]
predictions = db.get_predictions(user_id=user_id)
df = predictions_to_dataframe(predictions)

if df.empty:
    st.info("No prediction data yet — generate some predictions first.")
    st.stop()

df["Province"] = df["City"].apply(config.get_province_for_city)

def chart(title, fig):
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)

tabs = st.tabs(["Overview", "Conditions", "Behavior & Outcomes", "Time Trends", "Geography"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        chart("Severity Distribution", px.pie(
            df, names="Predicted Severity", color="Predicted Severity",
            color_discrete_map=config.SEVERITY_COLORS, hole=0.4,
        ))
    with c2:
        chart("Model Usage", px.pie(df, names="Model Used", hole=0.4))

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        chart("Weather Analysis", px.histogram(
            df, x="Weather", color="Predicted Severity", barmode="group",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
        chart("Road Condition Analysis", px.histogram(
            df, x="Road Condition", color="Predicted Severity", barmode="group",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
    with c2:
        chart("Speed Limit Distribution", px.box(
            df, x="Predicted Severity", y="Speed Limit", color="Predicted Severity",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
        chart("Traffic Density Analysis", px.histogram(
            df, x="Traffic Density", color="Predicted Severity", barmode="group",
            color_discrete_map=config.SEVERITY_COLORS,
        ))

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        chart("Vehicle Type Analysis", px.histogram(
            df, x="Vehicle Type", color="Predicted Severity", barmode="group",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
        chart("Traffic Violation Analysis", px.histogram(
            df, x="Traffic Violation", color="Predicted Severity", barmode="group",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
    with c2:
        chart("Seatbelt/Helmet Usage", px.histogram(
            df, x="Seatbelt/Helmet", color="Predicted Severity", barmode="group",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
        inj_fat = df.melt(value_vars=["Injuries", "Fatalities"], var_name="Type", value_name="Count")
        chart("Injury vs Fatality Distribution", px.histogram(inj_fat, x="Count", color="Type", barmode="overlay"))

with tabs[3]:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Month"] = d["Date"].dt.month_name()
    d["Day of Week"] = d["Date"].dt.day_name()
    d["Hour"] = pd.to_datetime(d["Time"].astype(str), errors="coerce").dt.hour

    c1, c2 = st.columns(2)
    with c1:
        chart("Monthly Trends", px.histogram(
            d, x="Month", color="Predicted Severity", barmode="stack",
            category_orders={"Month": config.MONTH_OPTIONS}, color_discrete_map=config.SEVERITY_COLORS,
        ))
        chart("Hourly Trends", px.histogram(
            d, x="Hour", color="Predicted Severity", barmode="stack",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
    with c2:
        chart("Day of Week Trends", px.histogram(
            d, x="Day of Week", color="Predicted Severity", barmode="stack",
            category_orders={"Day of Week": config.DAY_OF_WEEK_OPTIONS}, color_discrete_map=config.SEVERITY_COLORS,
        ))

with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        chart("Province-wise Comparison", px.histogram(
            df, x="Province", color="Predicted Severity", barmode="stack",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
    with c2:
        chart("City-wise Comparison", px.histogram(
            df, x="City", color="Predicted Severity", barmode="stack",
            color_discrete_map=config.SEVERITY_COLORS,
        ))
    chart("Top Accident Locations", px.bar(
        df["City"].value_counts().reset_index().rename(columns={"count": "Count", "City": "City"}).head(10),
        x="City", y="Count",
    ))
