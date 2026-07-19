"""
generate_sample_dataset.py
Optional helper: generates a synthetic accident dataset matching the schema
this app expects, so you can test the full pipeline (Dataset Management ->
Train Models -> Predict) without needing your real Punjab_Traffic_Accidents.csv
right away. Run:  python generate_sample_dataset.py
"""

import random
import numpy as np
import pandas as pd

import config

random.seed(42)
np.random.seed(42)

N_ROWS = 5000


def generate_row():
    weather = random.choices(config.WEATHER_OPTIONS, weights=[45, 20, 10, 5, 5, 15])[0]
    road_condition = random.choices(config.ROAD_CONDITION_OPTIONS, weights=[50, 25, 5, 10, 10])[0]
    light_condition = random.choice(config.LIGHT_CONDITION_OPTIONS)
    traffic_density = random.choice(config.TRAFFIC_DENSITY_OPTIONS)
    speed_limit = random.choice([30, 40, 60, 80, 100, 120])
    road_type = random.choice(config.ROAD_TYPE_OPTIONS)
    vehicle_type = random.choices(
        config.VEHICLE_TYPE_OPTIONS, weights=[35, 30, 10, 10, 10, 5]
    )[0]
    traffic_violation = random.choices(
        config.TRAFFIC_VIOLATION_OPTIONS, weights=[50, 20, 10, 10, 10]
    )[0]
    seatbelt = random.choices(config.SEATBELT_HELMET_OPTIONS, weights=[60, 40])[0]
    injuries = int(np.random.poisson(1.1))
    fatalities = int(np.random.poisson(0.15))
    hour = random.randint(0, 23)
    part_of_day = next(
        label for lo, hi, label in
        [(5, 11, "Morning"), (12, 16, "Afternoon"), (17, 20, "Evening"), (21, 23, "Night"), (0, 4, "Night")]
        if lo <= hour <= hi
    )
    day_of_week = random.choice(config.DAY_OF_WEEK_OPTIONS)
    month = random.choice(config.MONTH_OPTIONS)
    city = random.choice(list(config.CITY_COORDS.keys()))
    lat, lon = config.CITY_COORDS[city]
    lat += np.random.normal(0, 0.05)
    lon += np.random.normal(0, 0.05)

    # A rough, engineered severity score so the target is learnable
    score = (
        fatalities * 3
        + injuries * 1.2
        + (speed_limit >= 100) * 2
        + (traffic_violation != "None") * 2
        + (seatbelt == "No") * 1.5
        + (road_condition in ("Wet", "Icy", "Damaged")) * 1
        + (light_condition in ("Dark - Unlit",)) * 1
    )
    severity = "High" if score >= 6 else ("Medium" if score >= 2.5 else "Low")

    date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=random.randint(0, 730))

    return dict(
        Date=date.date().isoformat(),
        Weather=weather,
        Road_Condition=road_condition,
        Light_Condition=light_condition,
        Traffic_Density=traffic_density,
        Speed_Limit=speed_limit,
        Road_Type=road_type,
        Vehicle_Type=vehicle_type,
        Traffic_Violation=traffic_violation,
        Seatbelt_or_Helmet=seatbelt,
        Injuries=injuries,
        Fatalities=fatalities,
        Severity=severity,
        Hour=hour,
        Part_of_Day=part_of_day,
        Day_of_Week=day_of_week,
        Month=month,
        City=city,
        Latitude=round(lat, 6),
        Longitude=round(lon, 6),
    )


if __name__ == "__main__":
    rows = [generate_row() for _ in range(N_ROWS)]
    df = pd.DataFrame(rows)
    out_path = "sample_traffic_accidents.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["Severity"].value_counts())
