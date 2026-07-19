"""
config.py
Central configuration for the Traffic Accident Analysis & Severity Prediction System.
Holds paths, constants, feature lists, and the CITY_COORDS lookup table used
across the app for map plotting and auto-filling latitude/longitude.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
BACKUP_DIR = os.path.join(BASE_DIR, "database", "backups")

for _d in (DATASET_DIR, MODELS_DIR, ASSETS_DIR, BACKUP_DIR):
    os.makedirs(_d, exist_ok=True)

# Default SQLite DB (works out of the box). To use MySQL instead, set the
# DATABASE_URL environment variable, e.g.:
#   mysql+pymysql://user:password@localhost:3306/traffic_accidents
DATABASE_URL = os.environ.get(
    "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'database', 'app.db')}"
)

PROCESSED_DATASET_PATH = os.path.join(DATASET_DIR, "processed_dataset.csv")
RAW_DATASET_PATH = os.path.join(DATASET_DIR, "raw_dataset.csv")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.joblib")
BEST_MODEL_META_PATH = os.path.join(MODELS_DIR, "best_model_meta.joblib")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
APP_NAME = "Traffic Accident Analysis & Severity Prediction System"
APP_ICON = "🚦"
SESSION_TIMEOUT_MINUTES = 60

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# Columns used strictly for prediction (fed to the ML model)
FEATURE_COLUMNS = [
    "Weather",
    "Road_Condition",
    "Light_Condition",
    "Traffic_Density",
    "Speed_Limit",
    "Road_Type",
    "Vehicle_Type",
    "Traffic_Violation",
    "Seatbelt_or_Helmet",
    "Injuries",
    "Fatalities",
    "Hour",
    "Part_of_Day",
    "Day_of_Week",
    "Month",
]

# Columns kept only for storage / visualization / reporting - NEVER fed to the model
CONTEXT_COLUMNS = ["Date", "City", "Latitude", "Longitude"]

TARGET_COLUMN = "Severity"

CATEGORICAL_FEATURES = [
    "Weather",
    "Road_Condition",
    "Light_Condition",
    "Traffic_Density",
    "Road_Type",
    "Vehicle_Type",
    "Traffic_Violation",
    "Seatbelt_or_Helmet",
    "Part_of_Day",
    "Day_of_Week",
    "Month",
]

NUMERIC_FEATURES = ["Speed_Limit", "Injuries", "Fatalities", "Hour"]

SEVERITY_LEVELS = ["Low", "Medium", "High"]
SEVERITY_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

# Dropdown option sets (used both for synthetic data generation compatibility
# and for the Streamlit prediction form)
WEATHER_OPTIONS = ["Clear", "Rain", "Fog", "Dust Storm", "Snow", "Cloudy"]
ROAD_CONDITION_OPTIONS = ["Dry", "Wet", "Icy", "Under Construction", "Damaged"]
LIGHT_CONDITION_OPTIONS = ["Daylight", "Dusk/Dawn", "Dark - Lit", "Dark - Unlit"]
TRAFFIC_DENSITY_OPTIONS = ["Low", "Medium", "High"]
ROAD_TYPE_OPTIONS = ["Motorway", "Highway", "Urban Road", "Rural Road", "Intersection"]
VEHICLE_TYPE_OPTIONS = ["Car", "Motorcycle", "Bus", "Truck", "Rickshaw", "Van"]
TRAFFIC_VIOLATION_OPTIONS = ["None", "Overspeeding", "Signal Jump", "Wrong Way", "Illegal Overtaking"]
SEATBELT_HELMET_OPTIONS = ["Yes", "No"]
PART_OF_DAY_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
DAY_OF_WEEK_OPTIONS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_OPTIONS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ---------------------------------------------------------------------------
# Province -> City map & coordinates (approximate city-center lat/lon)
# ---------------------------------------------------------------------------
PROVINCE_CITIES = {
    'Punjab': ['Attock', 'Fateh Jang', 'Hasan Abdal', 'Hazro', 'Jand', 'Pindi Gheb','Bahawalnagar', 'Chishtian', 'Fort Abbas', 'Haroonabad','Bahawalpur', 'Hasilpur','Faisalabad City', 'Faisalabad Sadar', 'Jaranwala', 'Summundri', 'Tandlian Wala','D.G Khan (Tribal Area)', 'Dera Ghazi Khan', 'Kot Chatta','Gujranwala', 'Kamoke', 'Nowshera Virkan', 'Wazirabad','Gujrat','Jhang','Jhelum','Kasur','Jahanian', 'Kabirwala', 'Khanewal', 'Mian Channu','Khushab', 'Naushera','Hafizabad', 'Pindi Bhattian','Lahore Cantt', 'Lahore City','Leiah (Layyah)','Dunyapur', 'Kahror Pacca', 'Lodhran','Isakhel', 'Mianwali', 'Piplan','Multan City', 'Multan Saddar','Muzaffargarh','Nankana Sahib', 'Sangla Hill', 'Shah Kot','Liaqatpur', 'Rahim Yar Khan','Gujar Khan', 'Kahuta', 'Kallar Sayaddan', 'Kotli Sattian', 'Murree', 'Rawalpindi', 'Taxila'],
    'Sindh': ['Karachi', 'Hyderabad', 'Sukkur', 'Larkana', 'Nawabshah', 'Mirpur Khas', 'Jacobabad', 'Shikarpur', 'Khairpur', 'Dadu'],
    'KPK': ['Peshawar', 'Mardan', 'Mingora', 'Kohat', 'Abbottabad', 'Dera Ismail Khan', 'Swabi', 'Nowshera', 'Charsadda', 'Mansehra'],
    'Balochistan': ['Quetta', 'Gwadar', 'Khuzdar', 'Turbat', 'Chaman', 'Sibi', 'Zhob', 'Loralai', 'Hub', 'Dera Murad Jamali'],
}

CITY_COORDS = {
    # Punjab
    "Attock": (33.8165, 72.2995),
    "Fateh Jang": (33.5330, 72.6745),
    "Hasan Abdal": (33.8245, 72.7165),
    "Hazro": (33.9160, 72.4745),
    "Jand": (33.3330, 72.0495),
    "Pindi Gheb": (33.2665, 72.2910),

    "Bahawalnagar": (29.9930, 73.2520),
    "Chishtian": (29.7995, 72.8495),
    "Fort Abbas": (29.1745, 72.7745),
    "Haroonabad": (29.5245, 73.1330),

    "Bahawalpur": (29.3995, 71.6745),
    "Hasilpur": (29.6495, 72.5410),

    "Faisalabad City": (31.4245, 73.0910),
    "Faisalabad Sadar": (31.3995, 73.0830),
    "Jaranwala": (31.3410, 73.4495),
    "Summundri": (31.1165, 72.9830),
    "Tandlian Wala": (31.0830, 73.2830),

    "D.G Khan (Tribal Area)": (30.2160, 70.1830),
    "Dera Ghazi Khan": (30.0830, 70.6245),
    "Kot Chatta": (29.7995, 70.6245),

    "Gujranwala": (32.1915, 74.2000),
    "Kamoke": (31.9995, 74.2495),
    "Nowshera Virkan": (32.0165, 73.9080),
    "Wazirabad": (32.4330, 74.1995),

    "Gujrat": (32.5745, 74.1495),

    "Jhang": (31.3245, 72.2745),

    "Jhelum": (32.9660, 73.6995),

    "Kasur": (31.0995, 74.3915),

    "Jahanian": (30.2410, 71.9830),
    "Kabirwala": (30.5330, 71.8160),
    "Khanewal": (30.3165, 72.0415),
    "Mian Channu": (30.4940, 72.3830),

    "Khushab": (32.3330, 72.4165),
    "Naushera": (32.5495, 72.1830),

    "Hafizabad": (32.1330, 73.6745),
    "Pindi Bhattian": (31.9245, 73.3660),

    "Lahore Cantt": (31.5160, 74.4500),
    "Lahore City": (31.5580, 74.3160),

    "Leiah (Layyah)": (30.9245, 71.0915),

    "Dunyapur": (29.8495, 71.9415),
    "Kahror Pacca": (29.6160, 71.9830),
    "Lodhran": (29.5415, 71.6495),

    "Isakhel": (32.8830, 71.3330),
    "Mianwali": (32.6910, 71.5830),
    "Piplan": (32.3330, 71.3830),

    "Multan City": (30.1995, 71.4910),
    "Multan Saddar": (30.1080, 71.4745),

    "Muzaffargarh": (30.0660, 71.1745),

    "Nankana Sahib": (31.4245, 73.7495),
    "Sangla Hill": (31.7160, 73.7160),
    "Shah Kot": (31.5665, 73.7745),

    "Liaqatpur": (28.8665, 71.0415),
    "Rahim Yar Khan": (28.3660, 70.3660),

    "Gujar Khan": (33.2160, 73.3660),
    "Kahuta": (33.5330, 73.4830),
    "Kallar Sayaddan": (33.4160, 73.3830),
    "Kotli Sattian": (33.7495, 73.5245),
    "Murree": (33.9160, 73.4500),
    "Rawalpindi": (33.5995, 73.0415),
    "Taxila": (33.7665, 72.8160),

    # Sindh
    "Karachi": (24.9467, 67.0250),
    "Hyderabad": (25.3950, 68.3650),
    "Sukkur": (27.7000, 68.8400),
    "Larkana": (27.5450, 68.2100),
    "Nawabshah": (26.2400, 68.4100),
    "Mirpur Khas": (25.5200, 69.0000),
    "Jacobabad": (28.2850, 68.4400),
    "Shikarpur": (27.9550, 68.6400),
    "Khairpur": (27.5250, 68.7550),
    "Dadu": (26.7300, 67.7800),

    # KPK
    "Peshawar": (34.0050, 71.5350),
    "Mardan": (34.2050, 72.0600),
    "Mingora": (34.7800, 72.3600),
    "Kohat": (33.5750, 71.4600),
    "Abbottabad": (34.1600, 73.2300),
    "Dera Ismail Khan": (31.8200, 70.9000),
    "Swabi": (34.1300, 72.4600),
    "Nowshera": (33.9950, 71.9650),
    "Charsadda": (34.1550, 71.7400),
    "Mansehra": (34.3400, 73.2050),

    # Balochistan
    "Quetta": (30.2000, 67.0000),
    "Gwadar": (25.1400, 62.3750),
    "Khuzdar": (27.8000, 66.6100),
    "Turbat": (26.0000, 63.0600),
    "Chaman": (30.9250, 66.4400),
    "Sibi": (29.5500, 67.8800),
    "Zhob": (31.3400, 69.4550),
    "Loralai": (30.3700, 68.6050),
    "Hub": (25.0250, 66.8900),
    "Dera Murad Jamali": (28.5450, 68.2050),
}


def get_province_for_city(city: str) -> str:
    """Return the province a given city belongs to."""
    for province, cities in PROVINCE_CITIES.items():
        if city in cities:
            return province
    return "Unknown"