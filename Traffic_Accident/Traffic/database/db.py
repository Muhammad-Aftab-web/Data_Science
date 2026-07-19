"""
database/db.py
Engine/session management + reusable CRUD helper functions for Users and Predictions.
Using SQLAlchemy keeps the code portable between SQLite (default, zero-config)
and MySQL (set DATABASE_URL env var to a mysql+pymysql:// connection string).
"""

import shutil
import datetime as dt
from contextlib import contextmanager

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.models import Base, User, Prediction

_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------
def create_user(full_name, email, username, password_hash, role="Researcher/User"):
    with get_session() as s:
        user = User(
            full_name=full_name,
            email=email,
            username=username,
            password_hash=password_hash,
            role=role,
        )
        s.add(user)
        s.flush()
        return user.user_id


def get_user_by_username(username):
    with get_session() as s:
        user = s.query(User).filter(User.username == username).first()
        if user:
            s.expunge(user)
        return user


def get_user_by_email(email):
    with get_session() as s:
        user = s.query(User).filter(User.email == email).first()
        if user:
            s.expunge(user)
        return user


def get_all_users():
    with get_session() as s:
        users = s.query(User).order_by(User.created_at.desc()).all()
        for u in users:
            s.expunge(u)
        return users


def delete_user(user_id):
    with get_session() as s:
        user = s.query(User).filter(User.user_id == user_id).first()
        if user:
            s.delete(user)


def count_users():
    with get_session() as s:
        return s.query(func.count(User.user_id)).scalar()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def save_prediction(**kwargs) -> int:
    with get_session() as s:
        pred = Prediction(**kwargs)
        s.add(pred)
        s.flush()
        return pred.prediction_id


def get_predictions(user_id=None, filters=None, limit=None):
    """
    Fetch predictions. If user_id is None, returns predictions for ALL users
    (admin view). `filters` is an optional dict with keys: city, severity,
    date_from, date_to, weather.
    """
    filters = filters or {}
    with get_session() as s:
        q = s.query(Prediction)
        if user_id is not None:
            q = q.filter(Prediction.user_id == user_id)
        if filters.get("city"):
            q = q.filter(Prediction.city == filters["city"])
        if filters.get("severity"):
            q = q.filter(Prediction.predicted_severity == filters["severity"])
        if filters.get("weather"):
            q = q.filter(Prediction.weather == filters["weather"])
        if filters.get("date_from"):
            q = q.filter(Prediction.prediction_date >= filters["date_from"])
        if filters.get("date_to"):
            q = q.filter(Prediction.prediction_date <= filters["date_to"])
        q = q.order_by(Prediction.created_at.desc())
        if limit:
            q = q.limit(limit)
        results = q.all()
        for r in results:
            s.expunge(r)
        return results


def delete_prediction(prediction_id, user_id=None):
    """Delete a prediction. If user_id is given, only deletes if it belongs to that user."""
    with get_session() as s:
        q = s.query(Prediction).filter(Prediction.prediction_id == prediction_id)
        if user_id is not None:
            q = q.filter(Prediction.user_id == user_id)
        pred = q.first()
        if pred:
            s.delete(pred)
            return True
        return False


def reset_prediction_history():
    with get_session() as s:
        s.query(Prediction).delete()


def prediction_counts_by_severity(user_id=None):
    with get_session() as s:
        q = s.query(Prediction.predicted_severity, func.count(Prediction.prediction_id))
        if user_id is not None:
            q = q.filter(Prediction.user_id == user_id)
        q = q.group_by(Prediction.predicted_severity)
        return dict(q.all())


# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------
def backup_database():
    """Simple file-copy backup (SQLite only). Returns backup path."""
    if not config.DATABASE_URL.startswith("sqlite"):
        raise RuntimeError("Backup helper only supports the default SQLite database.")
    db_path = config.DATABASE_URL.replace("sqlite:///", "")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(config.BACKUP_DIR, f"app_backup_{ts}.db")
    shutil.copy2(db_path, backup_path)
    return backup_path


def restore_database(backup_path):
    if not config.DATABASE_URL.startswith("sqlite"):
        raise RuntimeError("Restore helper only supports the default SQLite database.")
    db_path = config.DATABASE_URL.replace("sqlite:///", "")
    shutil.copy2(backup_path, db_path)


def list_backups():
    if not os.path.isdir(config.BACKUP_DIR):
        return []
    return sorted(os.listdir(config.BACKUP_DIR), reverse=True)
