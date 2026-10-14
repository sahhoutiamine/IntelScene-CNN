# Beginner-friendly CPU environment for TensorFlow / Keras.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /workspace

# Install Python packages when requirements change (better build caching).
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Project source is mounted by docker-compose for easy editing.
CMD ["sleep", "infinity"]
