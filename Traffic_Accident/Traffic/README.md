# 🚦 AI-Powered Traffic Accident Analysis & Severity Prediction System

A full-stack Streamlit application for Pakistan-wide traffic accident severity
prediction, explainable AI, geographic visualization, analytics, and
reporting. Built for a Final Year Project, covering Punjab, Sindh, Khyber
Pakhtunkhwa, and Balochistan.

---

## ✨ Features

- **Authentication** — registration, login/logout, bcrypt password hashing,
  session timeout, role-based access (Administrator / Researcher-User).
  The first registered account automatically becomes Administrator.
- **Dashboard** — live metrics, accident map preview, severity breakdown,
  recent predictions, trend chart.
- **Dataset Management** — upload CSV, preview, stats, missing-value
  handling, save processed dataset. Only the approved feature columns are
  ever used for model training; `Date`, `City`, `Latitude`, `Longitude` are
  kept strictly for storage/visualization/reporting.
- **Machine Learning** — trains & compares Random Forest, XGBoost, LightGBM,
  Logistic Regression, Decision Tree, and an MLP Neural Network. Evaluates
  Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix and
  auto-selects the best model (saved with Joblib).
- **Prediction** — form-based input, auto-derives Hour/Part_of_Day/
  Day_of_Week/Month from Date+Time, returns predicted severity, class
  probabilities, confidence score, a SHAP-based explanation, and rule-based
  safety recommendations. Every prediction is saved to the database.
- **Prediction History** — search, filter (city/severity/date), paginate,
  delete, export to CSV/PDF. Admins see every user's history.
- **Analytics Dashboard** — 15+ interactive Plotly charts (weather, road
  condition, vehicle type, traffic density, violations, seatbelt/helmet use,
  injuries/fatalities, monthly/day-of-week/hourly trends, province & city
  comparisons, top locations).
- **Accident Map** — Folium map with marker clustering, heatmap toggle,
  color-coded severity markers, and province/city/severity/weather filters.
- **Reports** — downloadable CSV and PDF reports (per-prediction with SHAP,
  and full-history summaries).
- **Settings (Admin only)** — retrain models, switch the active model,
  upload a new dataset, backup/restore the database, reset prediction
  history, manage users.
- **UI extras** — dark/light theme toggle, toast notifications, pagination,
  search, download buttons.

---

## 🧱 Tech Stack

| Layer          | Technology                                  |
|----------------|----------------------------------------------|
| Frontend       | Streamlit                                     |
| ML             | Scikit-learn, XGBoost, LightGBM               |
| Explainability | SHAP                                          |
| Database       | SQLite (default, zero-config) or MySQL        |
| ORM            | SQLAlchemy                                    |
| Maps           | Folium + streamlit-folium                     |
| Charts         | Plotly                                        |
| Reports        | ReportLab (PDF), pandas (CSV)                 |
| Auth           | bcrypt password hashing + Streamlit session   |

---

## 📁 Project Structure

```
traffic_accident_system/
├── app.py                     # Entry point: login/registration/home
├── config.py                  # Paths, feature schema, CITY_COORDS, constants
├── requirements.txt
├── generate_sample_dataset.py # Optional: creates a synthetic test CSV
├── authentication/
│   └── auth.py                # bcrypt hashing, session mgmt, RBAC
├── database/
│   ├── models.py               # SQLAlchemy User & Prediction models
│   └── db.py                   # Engine/session + CRUD helpers, backup/restore
├── machine_learning/
│   ├── preprocessing.py        # Cleaning, encoding, train/test split
│   ├── training.py             # Multi-model training, evaluation, selection
│   └── predict.py              # Inference + SHAP + safety recommendations
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Dataset_Management.py
│   ├── 3_Train_Models.py
│   ├── 4_Predict.py
│   ├── 5_Prediction_History.py
│   ├── 6_Analytics.py
│   ├── 7_Accident_Map.py
│   ├── 8_Reports.py
│   └── 9_Settings.py           # Admin-only
├── reports/
│   └── report_generator.py     # PDF/CSV builders (ReportLab)
├── utils/
│   └── helpers.py               # Theming, badges, pagination, DataFrame helpers
├── assets/                     # Static assets (logos, etc.)
├── dataset/                    # Uploaded/processed CSVs land here
└── trained_models/             # Saved .joblib models + metadata
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Generate a sample dataset to test with

If you don't want to use your real `Punjab_Traffic_Accidents.csv` right away:

```bash
python generate_sample_dataset.py
```

This creates `sample_traffic_accidents.csv` (5,000 rows) matching the exact
schema the app expects.

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### 4. First-time workflow

1. **Register** an account on the Home page — the very first account
   automatically becomes **Administrator**.
2. Go to **Dataset Management** → upload your CSV (or the generated sample)
   → review stats → click **Handle Missing Values & Save Processed Dataset**.
3. Go to **Train Models** → click **Train & Compare All Models**. The best
   model (by macro F1) is automatically saved as the active model.
4. Go to **Predict** → fill in the form → get severity, confidence, SHAP
   explanation, and safety tips. The prediction is saved automatically.
5. Explore **Dashboard**, **Prediction History**, **Analytics**, and
   **Accident Map**.
6. As Administrator, use **Settings** to retrain, switch models, manage
   users, or backup/restore the database.

---

## 🗄️ Using MySQL instead of SQLite

By default the app uses a local SQLite file at `database/app.db` — no setup
needed. To use MySQL instead, set an environment variable before launching:

```bash
export DATABASE_URL="mysql+pymysql://<user>:<password>@<host>:3306/<db_name>"
streamlit run app.py
```

Make sure the target MySQL database already exists; SQLAlchemy will create
the tables automatically on first run.

> Note: the Backup/Restore feature in Settings currently only supports the
> default SQLite mode (simple file copy). For MySQL, use `mysqldump`/
> `mysql` directly, or extend `database/db.py`.

---

## 📊 Dataset Schema

| Column              | Used as Feature? | Notes                                  |
|---------------------|:---:|------------------------------------------------|
| Date                | ❌ | Storage / map / report only                     |
| Weather             | ✅ | Categorical                                      |
| Road_Condition      | ✅ | Categorical                                      |
| Light_Condition     | ✅ | Categorical                                      |
| Traffic_Density     | ✅ | Categorical                                      |
| Speed_Limit         | ✅ | Numeric                                          |
| Road_Type           | ✅ | Categorical                                      |
| Vehicle_Type        | ✅ | Categorical                                      |
| Traffic_Violation   | ✅ | Categorical                                      |
| Seatbelt_or_Helmet  | ✅ | Categorical                                      |
| Injuries            | ✅ | Numeric                                          |
| Fatalities          | ✅ | Numeric                                          |
| Severity            | 🎯 | Target (Low / Medium / High)                     |
| Hour                | ✅ | Numeric, derivable from Date+Time                |
| Part_of_Day         | ✅ | Categorical, derivable from Hour                 |
| Day_of_Week         | ✅ | Categorical, derivable from Date                 |
| Month               | ✅ | Categorical, derivable from Date                 |
| City                | ❌ | Storage / map / report only                      |
| Latitude            | ❌ | Storage / map / report only (from `CITY_COORDS`) |
| Longitude           | ❌ | Storage / map / report only (from `CITY_COORDS`) |

---

## 🔐 Security Notes

- Passwords are hashed with **bcrypt** (never stored in plaintext).
- All database access goes through **SQLAlchemy ORM** (parameterized
  queries — no raw SQL string concatenation, mitigating SQL injection).
- Sessions expire after `SESSION_TIMEOUT_MINUTES` (default 60) of
  inactivity, configurable in `config.py`.
- Role-based checks (`require_login`, `require_admin`) guard every page.

---

## ⚠️ Known Limitations / Things to Customize for Submission

- `CITY_COORDS` currently ships with ~23 major cities. Extend it in
  `config.py` if your dataset covers more.
- SHAP uses `TreeExplainer`, which works best with tree-based models
  (Random Forest, XGBoost, LightGBM, Decision Tree). For Logistic
  Regression/MLP, the app falls back to global feature importances if
  SHAP's tree explainer doesn't apply.
- The MySQL backup/restore flow is intentionally left simple (SQLite file
  copy) — swap in `mysqldump` for production MySQL use.
- No SMTP/email verification is implemented for registration — add one if
  your evaluators expect it.

---

## 👤 Credits

Developed as part of a Final Year Project — *Traffic Accident Analysis and
Severity Prediction System* — Department of Data Science.
