"""
utils/viz_utils.py
Reusable Plotly chart builders shared by the dashboard's "Custom
Visualizations" section (and any other page that wants ad-hoc charts).
"""

from typing import Optional

import pandas as pd
import plotly.express as px


def get_lat_lon(df: pd.DataFrame, config) -> pd.DataFrame:
    """Ensure the dataframe has Latitude/Longitude columns.

    If they're missing but a City column is present, coordinates are looked
    up from config.CITY_COORDS. Rows for unknown cities get NaN and are
    dropped later at plot time.
    """
    df = df.copy()
    has_coords = {"Latitude", "Longitude"}.issubset(df.columns)

    if not has_coords and "City" in df.columns:
        coords = df["City"].map(config.CITY_COORDS)
        df["Latitude"] = coords.map(lambda c: c[0] if isinstance(c, tuple) else None)
        df["Longitude"] = coords.map(lambda c: c[1] if isinstance(c, tuple) else None)
    elif has_coords:
        # Coerce to numeric in case they came in as strings
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    return df


def generic_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: Optional[str],
    y: Optional[str] = None,
    color: Optional[str] = None,
    size: Optional[str] = None,
):
    """Build a Plotly figure for the requested chart_type.

    Returns None if the chosen columns don't make sense for that chart type
    (e.g. a Line Chart with no Y axis), so the caller can show a friendly
    "select valid columns" message instead of crashing.
    """
    if df.empty or x is None:
        return None

    try:
        if chart_type == "Bar Chart":
            return px.bar(df, x=x, y=y, color=color, barmode="group")

        if chart_type == "Line Chart":
            if y is None:
                return None
            data = df.sort_values(by=x)
            return px.line(data, x=x, y=y, color=color)

        if chart_type == "Scatter Plot":
            if y is None:
                return None
            return px.scatter(df, x=x, y=y, color=color, size=size)

        if chart_type == "Histogram":
            return px.histogram(df, x=x, color=color)

        if chart_type == "Box Plot":
            return px.box(df, x=x, y=y, color=color)

        if chart_type == "Violin Plot":
            return px.violin(df, x=x, y=y, color=color, box=True)

        if chart_type == "Heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                return None
            corr = numeric_df.corr()
            return px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )

        if chart_type == "Pie Chart":
            return px.pie(df, names=x, color=color)

        if chart_type == "Area Chart":
            if y is None:
                return None
            data = df.sort_values(by=x)
            return px.area(data, x=x, y=y, color=color)

        if chart_type == "Sunburst":
            path = [c for c in [x, color] if c]
            if not path:
                return None
            return px.sunburst(df, path=path, values=y if y else None)

        if chart_type == "Geographic Map":
            # Convention used by the Dashboard page: x = Longitude, y = Latitude
            if x not in df.columns or y not in (df.columns if y else []):
                return None
            map_df = df.dropna(subset=[x, y])
            if map_df.empty:
                return None
            fig = px.scatter_mapbox(
                map_df,
                lat=y,
                lon=x,
                color=color,
                size=size,
                zoom=4.2,
                height=520,
                mapbox_style="open-street-map",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            return fig

    except Exception:
        # Any column/type mismatch (e.g. numeric-only chart on text column)
        # is treated as "not renderable" rather than crashing the page.
        return None

    return None
