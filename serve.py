import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ============================================================
# Application Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://13.48.169.158:5001"
)

MODEL_URI = "models:/house-price-predictor@champion"


# ============================================================
# Model Features
# ============================================================

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# Load Model
# ============================================================

model = mlflow.sklearn.load_model(
    MODEL_URI
)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="House Price Predictor",
    description="MLflow + FastAPI House Price Prediction API",
    version="1.0.0"
)


# ============================================================
# Request Schema
# ============================================================

class HouseFeatures(BaseModel):

    sqft: float = Field(
        ...,
        gt=0,
        le=20000
    )

    bedrooms: int = Field(
        ...,
        gt=0,
        le=20
    )

    bathrooms: float = Field(
        ...,
        gt=0,
        le=20
    )

    age_years: int = Field(
        ...,
        ge=0,
        le=200
    )

    garage: int = Field(
        ...,
        ge=0,
        le=10
    )

    location_score: float = Field(
        ...,
        ge=1,
        le=10
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_URI,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }


# ============================================================
# Prediction API
# ============================================================

@app.post("/predict")
def predict(
    features: HouseFeatures
):

    input_data = features.model_dump()

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    prediction = model.predict(
        input_df
    )[0]

    return {
        "predicted_price": round(
            float(prediction),
            2
        )
    }


# ============================================================
# Static Frontend
# ============================================================

STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# Frontend
# ============================================================

@app.get("/")
def frontend():

    return FileResponse(
        STATIC_DIR / "index.html"
    )