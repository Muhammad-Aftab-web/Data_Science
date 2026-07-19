"""
app.py
Home page: login / registration. Streamlit's multipage-app mechanism will
automatically pick up every file inside pages/ and list it in the sidebar
once the user is authenticated.
"""

import streamlit as st

import config
from database import db
from authentication import auth
from utils.helpers import inject_theme_css, toast

st.set_page_config(page_title=config.APP_NAME, page_icon=config.APP_ICON, layout="wide")

db.init_db()

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
inject_theme_css(st.session_state["dark_mode"])


def render_login_form():
    with st.form("login_form"):
        username = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", use_container_width=True)
        if submitted:
            if not username or not password:
                st.error("Please fill in both fields.")
                return
            user, error = auth.login_user(username, password)
            if error:
                st.error(error)
            else:
                auth.start_session(user)
                toast(f"Welcome back, {user.full_name}!")
                st.rerun()


def render_register_form():
    with st.form("register_form"):
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        username = st.text_input("Choose a Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            errors = auth.validate_registration(
                full_name, email, username, password, confirm_password
            )
            if errors:
                for e in errors:
                    st.error(e)
            else:
                user_id, role = auth.register_user(full_name, email, username, password)
                toast(f"Account created as {role}! Please log in.")
                st.balloons()


def render_authenticated_home():
    st.title(f"{config.APP_ICON} {config.APP_NAME}")
    st.success(f"Logged in as **{st.session_state['auth_full_name']}** ({st.session_state['auth_role']})")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            """
            Use the sidebar to navigate:
            - **Dashboard** — key metrics & recent activity
            - **Dataset Management** — upload & preprocess accident data
            - **Train Models** — compare ML models and select the best
            - **Predict** — get a severity prediction with SHAP explanation
            - **Prediction History** — search, filter, export your predictions
            - **Analytics** — interactive Plotly dashboards
            - **Accident Map** — Folium map with clustering & heatmaps
            - **Reports** — download PDF/CSV reports
            - **Settings** — (Administrator) retrain models, manage data
            """
        )
    with col2:
        st.session_state["dark_mode"] = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"])
        if st.button("Log Out", use_container_width=True):
            auth.logout_user()
            st.rerun()

    st.divider()
    total_users = db.count_users()
    counts = db.prediction_counts_by_severity(
        user_id=None if auth.is_admin() else st.session_state["auth_user_id"]
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", total_users)
    c2.metric("High Severity", counts.get("High", 0))
    c3.metric("Medium Severity", counts.get("Medium", 0))
    c4.metric("Low Severity", counts.get("Low", 0))


if auth.is_authenticated():
    render_authenticated_home()
else:
    st.title(f"{config.APP_ICON} {config.APP_NAME}")
    st.caption("AI-powered accident severity prediction & analytics for Punjab, Sindh, KP, and Balochistan.")
    tab_login, tab_register = st.tabs(["🔑 Log In", "📝 Register"])
    with tab_login:
        render_login_form()
    with tab_register:
        render_register_form()
