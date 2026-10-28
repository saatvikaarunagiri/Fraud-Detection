# Image for the fraud services (scoring API, stream worker).
# Default REQS is the light set (serves the API); pass the full requirements
# for the Bytewax stream + Claude investigator.
#
#   docker build -t fraud-detection .
#   docker build -t fraud-full --build-arg REQS=p2_fraud/requirements.txt .

FROM python:3.12-slim

ARG REQS=requirements-ci.txt
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# libgomp1 for xgboost, curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt ./requirements-ci.txt
COPY p2_fraud/requirements.txt ./p2_fraud/requirements.txt
RUN pip install --upgrade pip && pip install -r ${REQS}

COPY . .

# Default: train a small model + serve the API. Compose/fly override as needed.
CMD ["sh", "-c", "python -m p2_fraud.data.generate --n 20000 && python -m p2_fraud.fraud.train && uvicorn p2_fraud.serve.app:app --host 0.0.0.0 --port 8000"]
