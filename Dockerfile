# Multi-stage build for production optimization
FROM nvidia/cuda:12.9.0-base-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    curl \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    uvicorn[standard] \
    psutil \
    prometheus_client \
    redis[hiredis] \
    accelerate

# Copy application code
COPY . .

# Create cache directories and set permissions
RUN mkdir -p /app/.cache/huggingface/hub && \
    mkdir -p /app/logs && \
    mkdir -p /app/embedding_cache && \
    mkdir -p /app/query_cache

# Create non-root user
RUN useradd --create-home --shell /bin/bash raptor && \
    chown -R raptor:raptor /app

USER raptor

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "generic-qa-server.py"]