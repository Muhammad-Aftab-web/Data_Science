import os
import datetime as dt

import streamlit as st
import plotly.express as px

import config
from authentication import auth
from database import db
from utils.helpers import inject_theme_css, toast, severity_badge
from machine_learning.predict import predict_severity, safety_recommendations
from reports.report_generator import build_single_prediction_pdf

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("🔮 Predict Accident Severity")

if not os.path.exists(config.BEST_MODEL_META_PATH):
    st.warning("No trained model found yet. Go to **Train Models** first.")
    st.stop()
st.subheader("📍 Location")

province = st.selectbox(
    "Province",
    list(config.PROVINCE_CITIES.keys())
)

city = st.selectbox(
    "City",
    config.PROVINCE_CITIES[province]
)
latitude = st.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    format="%.6f"
)

with st.form("prediction_form"):
    st.subheader("📍 Database Fields")
    c1, c2, c3 = st.columns(3)
    date = c1.date_input("Date", value=dt.date.today())
    time_ = c2.time_input("Time", value=dt.datetime.now().time())
    # province = c3.selectbox("Province", config.PROVINCE_CITIES.keys())
    # city = c3.selectbox("City", config.PROVINCE_CITIES[province])

    st.subheader("🧾 Prediction Features")
    c1, c2, c3 = st.columns(3)
    weather = c1.selectbox("Weather", config.WEATHER_OPTIONS)
    road_condition = c2.selectbox("Road Condition", config.ROAD_CONDITION_OPTIONS)
    light_condition = c3.selectbox("Light Condition", config.LIGHT_CONDITION_OPTIONS)

    c1, c2, c3 = st.columns(3)
    traffic_density = c1.selectbox("Traffic Density", config.TRAFFIC_DENSITY_OPTIONS)
    speed_limit = c2.slider("Speed Limit (km/h)", 20, 140, 60, step=5)
    road_type = c3.selectbox("Road Type", config.ROAD_TYPE_OPTIONS)

    c1, c2, c3 = st.columns(3)
    vehicle_type = c1.selectbox("Vehicle Type", config.VEHICLE_TYPE_OPTIONS)
    traffic_violation = c2.selectbox("Traffic Violation", config.TRAFFIC_VIOLATION_OPTIONS)
    seatbelt_or_helmet = c3.selectbox("Seatbelt / Helmet Worn", config.SEATBELT_HELMET_OPTIONS)

    c1, c2 = st.columns(2)
    injuries = c1.number_input("Number of Injuries", min_value=0, max_value=50, value=0)
    fatalities = c2.number_input("Number of Fatalities", min_value=0, max_value=20, value=0)

    submitted = st.form_submit_button("🔮 Predict Severity", type="primary", use_container_width=True)

if submitted:
    form_data = dict(
        weather=weather,
        road_condition=road_condition,
        light_condition=light_condition,
        traffic_density=traffic_density,
        speed_limit=speed_limit,
        road_type=road_type,
        vehicle_type=vehicle_type,
        traffic_violation=traffic_violation,
        seatbelt_or_helmet=seatbelt_or_helmet,
        injuries=injuries,
        fatalities=fatalities,
    )
    result = predict_severity(form_data, date, time_)
    st.session_state["last_prediction"] = result
    st.session_state["last_prediction_context"] = {
        "date": date, "time": time_, "city": city, "form_data": form_data,
    }

    lat, lon = config.CITY_COORDS.get(city, (30.3753, 69.3451))
    db.save_prediction(
        user_id=st.session_state["auth_user_id"],
        prediction_date=date,
        prediction_time=time_,
        city=city,
        latitude=latitude,
        longitude=longitude,
        # latitude=lat,
        # longitude=lon,
        weather=weather,
        road_condition=road_condition,
        light_condition=light_condition,
        traffic_density=traffic_density,
        speed_limit=speed_limit,
        road_type=road_type,
        vehicle_type=vehicle_type,
        traffic_violation=traffic_violation,
        seatbelt_or_helmet=seatbelt_or_helmet,
        injuries=injuries,
        fatalities=fatalities,
        predicted_severity=result["severity"],
        confidence_score=result["confidence"],
        model_used=result["model_used"],
    )
    toast("Prediction saved to history.")

result = st.session_state.get("last_prediction")
ctx = st.session_state.get("last_prediction_context")

if result:
    st.divider()
    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Predicted Severity**")
        st.markdown(severity_badge(result["severity"]), unsafe_allow_html=True)
    c2.metric("Confidence Score", f"{result['confidence'] * 100:.1f}%")
    c3.metric("Model Used", result["model_used"])

    if result["probabilities"]:
        prob_fig = px.bar(
            x=list(result["probabilities"].keys()),
            y=list(result["probabilities"].values()),
            color=list(result["probabilities"].keys()),
            color_discrete_map=config.SEVERITY_COLORS,
            labels={"x": "Severity", "y": "Probability"},
        )
        st.plotly_chart(prob_fig, use_container_width=True)

    if result["shap_top_features"]:
        st.subheader("🧠 Explainable AI (SHAP) — Top Contributing Factors")
        shap_fig = px.bar(
            x=[f["impact"] for f in result["shap_top_features"]],
            y=[f["feature"] for f in result["shap_top_features"]],
            orientation="h",
            labels={"x": "Relative Impact", "y": "Feature"},
        )
        shap_fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(shap_fig, use_container_width=True)

    st.subheader("🛡️ Safety Recommendations")
    for tip in safety_recommendations(result["feature_record"], result["severity"]):
        st.markdown(f"- {tip}")

    if ctx:
        pdf_bytes = build_single_prediction_pdf(
            user_full_name=st.session_state["auth_full_name"],
            prediction_dict=result,
            record=result["feature_record"],
            date=ctx["date"],
            time_=ctx["time"],
            city=ctx["city"],
        )
        st.download_button(
            "📄 Download PDF Report", data=pdf_bytes,
            file_name=f"prediction_report_{ctx['date']}.pdf", mime="application/pdf",
        )
