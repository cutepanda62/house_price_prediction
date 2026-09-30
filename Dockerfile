
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application
COPY serve.py .

# Copy frontend
COPY static ./static

# FastAPI port
EXPOSE 8000

# Default MLflow URL
ENV MLFLOW_TRACKING=http://host.docker.internal:5001

# Start FastAPI
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]