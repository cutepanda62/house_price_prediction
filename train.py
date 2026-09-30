import os
from io import StringIO

import boto3
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://13.48.169.158:5001"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "mlops-house-prediction"

mlflow.set_experiment(EXPERIMENT_NAME)

client = mlflow.MlflowClient()


# ============================================================
# AWS S3 Configuration
# ============================================================

BUCKET = "housepricepredictionmlops"

KEY = (
    "proccessed/2026-09-30/"
    "Mlops_house_prediction_clean_v1 (1).csv"
)

s3 = boto3.client("s3")


# ============================================================
# Fetch Data
# ============================================================

def fetch_data():

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=KEY
    )

    df = pd.read_csv(
        StringIO(
            obj["Body"].read().decode("utf-8")
        )
    )

    return df


df = fetch_data()

print(f"Fetched shape: {df.shape}")


# ============================================================
# Features and Target
# ============================================================

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]

TARGET = "price"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# MLflow Run
# ============================================================

with mlflow.start_run() as run:

    run_id = run.info.run_id

    # Model parameters
    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Log parameters
    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "data_source",
        f"s3://{BUCKET}/{KEY}"
    )

    mlflow.log_param(
        "features",
        ",".join(FEATURES)
    )

    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Register model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="house-price-predictor"
    )

    # Find registered version
    model_versions = client.search_model_versions(
        f"name='house-price-predictor' and run_id='{run_id}'"
    )

    if not model_versions:
        raise RuntimeError(
            "Registered model version was not found."
        )

    model_version = model_versions[0]

    # Set champion alias
    client.set_registered_model_alias(
        name="house-price-predictor",
        alias="champion",
        version=model_version.version
    )

    print("\n======================================")
    print("Training completed successfully")
    print("======================================")

    print(f"Run ID        : {run_id}")
    print("Model         : house-price-predictor")
    print(f"Version       : {model_version.version}")
    print("Alias         : champion")

    print(f"MAE           : {mae:.2f}")
    print(f"RMSE          : {rmse:.2f}")
    print(f"R2 Score      : {r2:.4f}")

    print(
        f"MLflow Server : {MLFLOW_TRACKING_URI}"
    )

    print("======================================")