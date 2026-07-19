"""
database/models.py
SQLAlchemy ORM models for Users and Predictions.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    Time,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="Researcher/User")  # Administrator | Researcher/User
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)

    prediction_date = Column(Date, nullable=False)
    prediction_time = Column(Time, nullable=False)

    city = Column(String(80), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    weather = Column(String(50))
    road_condition = Column(String(50))
    light_condition = Column(String(50))
    traffic_density = Column(String(20))
    speed_limit = Column(Integer)
    road_type = Column(String(50))
    vehicle_type = Column(String(50))
    traffic_violation = Column(String(50))
    seatbelt_or_helmet = Column(String(10))
    injuries = Column(Integer)
    fatalities = Column(Integer)

    predicted_severity = Column(String(20), nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    shap_explanation = Column(Text, nullable=True)  # JSON-serialized top features

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction {self.prediction_id} {self.city} {self.predicted_severity}>"
