"""
authentication/auth.py
Handles registration, login, logout, password hashing (bcrypt), and
Streamlit session-state based session management with a timeout.
"""

import re
import datetime as dt

import bcrypt
import streamlit as st

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database import db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_registration(full_name, email, username, password, confirm_password):
    errors = []
    if not full_name or len(full_name.strip()) < 3:
        errors.append("Full name must be at least 3 characters.")
    if not email or not EMAIL_RE.match(email):
        errors.append("Please enter a valid email address.")
    if not username or len(username.strip()) < 3:
        errors.append("Username must be at least 3 characters.")
    if not re.match(r"^[A-Za-z0-9_]+$", username or ""):
        errors.append("Username may only contain letters, numbers, and underscores.")
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    if db.get_user_by_username(username):
        errors.append("Username is already taken.")
    if db.get_user_by_email(email):
        errors.append("Email is already registered.")
    return errors


# ---------------------------------------------------------------------------
# Registration / login / logout
# ---------------------------------------------------------------------------
def register_user(full_name, email, username, password, role="Researcher/User"):
    password_hash = hash_password(password)
    # First-ever user automatically becomes Administrator
    if db.count_users() == 0:
        role = "Administrator"
    user_id = db.create_user(full_name, email, username, password_hash, role)
    return user_id, role


def login_user(username_or_email: str, password: str):
    user = db.get_user_by_username(username_or_email) or db.get_user_by_email(username_or_email)
    if user is None:
        return None, "No account found with that username/email."
    if not verify_password(password, user.password_hash):
        return None, "Incorrect password."
    return user, None


def start_session(user):
    st.session_state["auth_user_id"] = user.user_id
    st.session_state["auth_username"] = user.username
    st.session_state["auth_full_name"] = user.full_name
    st.session_state["auth_role"] = user.role
    st.session_state["auth_login_time"] = dt.datetime.now()


def logout_user():
    for key in [
        "auth_user_id",
        "auth_username",
        "auth_full_name",
        "auth_role",
        "auth_login_time",
    ]:
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    if "auth_user_id" not in st.session_state:
        return False
    login_time = st.session_state.get("auth_login_time")
    if login_time and (dt.datetime.now() - login_time) > dt.timedelta(
        minutes=config.SESSION_TIMEOUT_MINUTES
    ):
        logout_user()
        return False
    return True


def is_admin() -> bool:
    return st.session_state.get("auth_role") == "Administrator"


def require_login():
    """Call at the top of every page. Stops rendering if not logged in."""
    if not is_authenticated():
        st.warning("🔒 Please log in from the Home page to access this page.")
        st.stop()


def require_admin():
    require_login()
    if not is_admin():
        st.error("⛔ This page is restricted to Administrators only.")
        st.stop()
