import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

import config
from authentication import auth
from database import db
from utils.helpers import inject_theme_css, predictions_to_dataframe

st.set_page_config(page_title="Accident Map", page_icon="🗺️", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("🗺️ Pakistan Accident Map")

user_id = None if auth.is_admin() else st.session_state["auth_user_id"]

st.subheader("🔍 Filters")
c1, c2, c3, c4 = st.columns(4)
province_filter = c1.selectbox("Province", ["All"] + list(config.PROVINCE_CITIES.keys()))
city_options = (
    config.PROVINCE_CITIES[province_filter] if province_filter != "All" else sorted(config.CITY_COORDS.keys())
)
city_filter = c2.selectbox("City", ["All"] + city_options)
severity_filter = c3.selectbox("Severity", ["All"] + config.SEVERITY_LEVELS)
weather_filter = c4.selectbox("Weather", ["All"] + config.WEATHER_OPTIONS)

view_mode = st.radio("View Mode", ["Marker Clusters", "Heatmap"], horizontal=True)

filters = {}
if city_filter != "All":
    filters["city"] = city_filter
if severity_filter != "All":
    filters["severity"] = severity_filter
if weather_filter != "All":
    filters["weather"] = weather_filter

predictions = db.get_predictions(user_id=user_id, filters=filters)
if province_filter != "All" and city_filter == "All":
    predictions = [p for p in predictions if config.get_province_for_city(p.city) == province_filter]

st.caption(f"{len(predictions)} location(s) plotted.")

m = folium.Map(location=[30.3753, 69.3451], zoom_start=5.3, tiles="OpenStreetMap")

if predictions:
    if view_mode == "Marker Clusters":
        cluster = MarkerCluster().add_to(m)
        for p in predictions:
            color = {"Low": "green", "Medium": "orange", "High": "red"}.get(p.predicted_severity, "blue")
            folium.Marker(
                location=[p.latitude, p.longitude],
                popup=folium.Popup(
                    f"<b>{p.city}</b><br>Severity: {p.predicted_severity}<br>"
                    f"Weather: {p.weather}<br>Date: {p.prediction_date}<br>"
                    f"Model: {p.model_used}",
                    max_width=250,
                ),
                icon=folium.Icon(color=color, icon="car", prefix="fa"),
            ).add_to(cluster)
    else:
        heat_data = [[p.latitude, p.longitude, 1] for p in predictions]
        HeatMap(heat_data, radius=18, blur=15).add_to(m)
else:
    st.info("No predictions match the current filters yet.")

st_folium(m, use_container_width=True, height=560)

if predictions:
    st.subheader("📍 Province-wise Distribution")
    from collections import Counter
    province_counts = Counter(config.get_province_for_city(p.city) for p in predictions)
    st.bar_chart(province_counts)

    st.subheader("🏙️ City-wise Distribution")
    city_counts = Counter(p.city for p in predictions)
    st.bar_chart(city_counts)
