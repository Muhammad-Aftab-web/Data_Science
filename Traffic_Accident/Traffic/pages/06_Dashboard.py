import streamlit as st
import pandas as pd
import plotly.express as px

import config
from authentication import auth
from utils.helpers import inject_theme_css
from utils.viz_utils import generic_chart, get_lat_lon

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
auth.require_login()
inject_theme_css(st.session_state.get("dark_mode", False))

st.title("📊 Dashboard")
st.caption("Exploring the underlying dataset — not individual predictions.")


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df = load_dataset(config.PROCESSED_DATASET_PATH)
except FileNotFoundError:
    st.error(
        "No processed dataset found at "
        f"`{config.PROCESSED_DATASET_PATH}`. Upload / generate a dataset first."
    )
    st.stop()

if df.empty:
    st.info("The dataset is empty — nothing to visualize yet.")
    st.stop()

# Fill in Latitude/Longitude from the City column (via config.CITY_COORDS)
# whenever the dataset doesn't already carry coordinates.
df = get_lat_lon(df, config)

severity_col = config.TARGET_COLUMN if config.TARGET_COLUMN in df.columns else None
counts = df[severity_col].value_counts().to_dict() if severity_col else {}

# ---------------------------------------------------------------------------
# Top-line metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("🔴 High Severity", counts.get("High", 0))
c3.metric("🟠 Medium Severity", counts.get("Medium", 0))
c4.metric("🟢 Low Severity", counts.get("Low", 0))

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("🗺️ Accident Locations")
    if {"Latitude", "Longitude"}.issubset(df.columns):
        map_df = df.dropna(subset=["Latitude", "Longitude"])
        if map_df.empty:
            st.info("No rows with valid coordinates to plot.")
        else:
            hover_cols = [c for c in ["Date", "Weather", "City"] if c in map_df.columns]
            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color=severity_col if severity_col else None,
                color_discrete_map=config.SEVERITY_COLORS if severity_col else None,
                hover_name="City" if "City" in map_df.columns else None,
                hover_data=hover_cols,
                zoom=4.2,
                height=420,
            )
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No City / Latitude / Longitude columns found in the dataset.")

with right:
    st.subheader("📈 Severity Split")
    if severity_col:
        pie = px.pie(
            df,
            names=severity_col,
            color=severity_col,
            color_discrete_map=config.SEVERITY_COLORS,
            hole=0.45,
        )
        pie.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info(f"No `{config.TARGET_COLUMN}` column found in the dataset.")

st.divider()

# ---------------------------------------------------------------------------
# Sample records + time trend
# ---------------------------------------------------------------------------
st.subheader("🕘 Sample Records")
preview_cols = [c for c in ["Date", "City", "Weather", severity_col] if c and c in df.columns]
st.dataframe(
    df[preview_cols].head(10) if preview_cols else df.head(10),
    use_container_width=True,
    hide_index=True,
)

if "Date" in df.columns and severity_col:
    st.subheader("📅 Records Over Time")
    trend = df.copy()
    trend["Date"] = pd.to_datetime(trend["Date"], errors="coerce")
    trend = trend.dropna(subset=["Date"])
    if not trend.empty:
        daily = (
            trend.groupby([trend["Date"].dt.date, severity_col])
            .size()
            .reset_index(name="Count")
        )
        fig2 = px.bar(
            daily,
            x="Date",
            y="Count",
            color=severity_col,
            color_discrete_map=config.SEVERITY_COLORS,
            barmode="stack",
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Custom visualization builder — every chart type + geographic map
# ---------------------------------------------------------------------------
st.header("📊 Custom Visualizations")

chart_type = st.selectbox(
    "Select chart type",
    [
        "Bar Chart",
        "Line Chart",
        "Scatter Plot",
        "Histogram",
        "Box Plot",
    ],
)

fig3 = None

if chart_type == "Geographic Map":
    if {"Latitude", "Longitude"}.issubset(df.columns):
        col_a, col_b = st.columns(2)
        with col_a:
            color_by = st.selectbox("Color by", ["(none)"] + list(df.columns), key="map_color")
        with col_b:
            numeric_cols = list(df.select_dtypes("number").columns)
            size_by = st.selectbox("Size by (numeric, optional)", ["(none)"] + numeric_cols, key="map_size")

        fig3 = generic_chart(
            df,
            chart_type,
            x="Longitude",
            y="Latitude",
            color=None if color_by == "(none)" else color_by,
            size=None if size_by == "(none)" else size_by,
        )
    else:
        st.info("Dataset has no Latitude/Longitude (or City) columns to map.")
else:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        x_axis = st.selectbox("X-axis", options=df.columns, key="x_axis_select")
    with col_b:
        y_axis = st.selectbox("Y-axis (optional)", options=["(none)"] + list(df.columns), key="y_axis_select")
    with col_c:
        color_by = st.selectbox("Color grouping (optional)", options=["(none)"] + list(df.columns), key="color_select")

    y_val = None if y_axis == "(none)" else y_axis
    color_val = None if color_by == "(none)" else color_by

    fig3 = generic_chart(df, chart_type, x_axis, y_val, color_val)

if fig3 is not None:
    st.plotly_chart(fig3, use_container_width=True)
elif chart_type != "Geographic Map":
    st.info("Select valid columns to render this chart type.")

st.divider()
# ---------------------------------------------------------------------------
# Custom visualization builder — every chart type + geographic map
# ---------------------------------------------------------------------------
st.header("📊 Custom Visualizations")

chart_types = st.selectbox(
    "Select chart type",
    [
        "Violin Plot",
        "Heatmap",
        "Pie Chart",
        "Area Chart",
        "Sunburst",
        "Geographic Map",
    ],
)

fig4 = None

if chart_types == "Geographic Map":
    if {"Latitude", "Longitude"}.issubset(df.columns):
        col_d, col_e = st.columns(2)
        with col_d:
            color_by = st.selectbox("Color by", ["(none)"] + list(df.columns), key="map_color")
        with col_e:
            numeric_cols = list(df.select_dtypes("number").columns)
            size_by = st.selectbox("Size by (numeric, optional)", ["(none)"] + numeric_cols, key="map_size")

        fig4 = generic_chart(
            df,
            chart_type,
            x="Longitude",
            y="Latitude",
            color=None if color_by == "(none)" else color_by,
            size=None if size_by == "(none)" else size_by,
        )
    else:
        st.info("Dataset has no Latitude/Longitude (or City) columns to map.")
else:
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        x_axis_1 = st.selectbox("X-axis", options=df.columns, key="x_axis_select_1")
    with col_e:
        y_axis_1 = st.selectbox("Y-axis (optional)", options=["(none)"] + list(df.columns), key="y_axis_select_1")
    with col_f:
        color_by_1 = st.selectbox("Color grouping (optional)", options=["(none)"] + list(df.columns), key="color_select_1")

    y_val = None if y_axis_1 == "(none)" else y_axis_1
    color_val = None if color_by_1 == "(none)" else color_by_1

    fig4 = generic_chart(df, chart_types, x_axis_1, y_val, color_val)

if fig4 is not None:
    st.plotly_chart(fig4, use_container_width=True)
elif chart_types != "Geographic Map":
    st.info("Select valid columns to render this chart type.")
