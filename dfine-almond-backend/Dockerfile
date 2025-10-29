FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements-base.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY model.py _wsgi.py ./

# Create directory for model weights
RUN mkdir -p /app/models

# Expose port
EXPOSE 9091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:9091/health', timeout=5)"

# Run with gunicorn for production
CMD ["label-studio-ml", "start", "model", "--port", "9091", "--host", "0.0.0.0"]
