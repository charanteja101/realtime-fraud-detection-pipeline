FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# libgomp1: OpenMP runtime required by XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraudstream/ fraudstream/

# default command = the scoring API; compose overrides it for the other services
CMD ["uvicorn", "fraudstream.api:app", "--host", "0.0.0.0", "--port", "8000"]
