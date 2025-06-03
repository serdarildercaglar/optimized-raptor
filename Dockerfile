# Multi-stage build for production optimization with GPU support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as base

# Set environment variables for GPU and caching
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# HuggingFace optimizations
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub

# PyTorch optimizations
ENV TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6+PTX"
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    python3.11-venv \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    # GPU monitoring tools
    nvidia-utils-525 \
    # Cleanup tools
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Create cache directories first
RUN mkdir -p /app/.cache/huggingface/hub && \
    mkdir -p /app/.cache/huggingface/datasets && \
    mkdir -p /app/logs && \
    mkdir -p /app/embedding_cache && \
    mkdir -p /app/query_cache && \
    mkdir -p /app/.local_model_cache

# Install Python dependencies in stages for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install --no-cache-dir \
    transformers[torch] \
    sentence-transformers \
    accelerate \
    scipy \
    scikit-learn \
    numpy

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    uvicorn[standard] \
    psutil \
    prometheus_client \
    redis[hiredis] \
    python-multipart

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 raptor && \
    chown -R raptor:raptor /app

# Switch to non-root user
USER raptor

# Copy application code
COPY --chown=raptor:raptor . .

# Pre-warm Python imports (helps with startup time)
RUN python -c "
import torch
import transformers
import sentence_transformers
import numpy
import redis
import fastapi
import uvicorn
print('✅ Core imports successful')
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
"

# Add model pre-loading script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting RAPTOR with model pre-loading..."\n\
python -c "\n\
import os\n\
import sys\n\
import time\n\
import logging\n\
\n\
logging.basicConfig(level=logging.INFO)\n\
logger = logging.getLogger(__name__)\n\
\n\
def preload_models():\n\
    try:\n\
        logger.info('📥 Pre-loading sentence transformers...')\n\
        from sentence_transformers import SentenceTransformer\n\
        \n\
        # Common models used by RAPTOR\n\
        models = [\n\
            'intfloat/multilingual-e5-large',\n\
            'sentence-transformers/multi-qa-mpnet-base-cos-v1'\n\
        ]\n\
        \n\
        for model_name in models:\n\
            if os.path.exists(f'/app/.cache/huggingface/hub'):\n\
                logger.info(f'Loading {model_name}...')\n\
                try:\n\
                    model = SentenceTransformer(model_name)\n\
                    # Test encoding\n\
                    _ = model.encode('test')\n\
                    logger.info(f'✅ {model_name} loaded successfully')\n\
                    del model\n\
                except Exception as e:\n\
                    logger.warning(f'⚠️ Failed to load {model_name}: {e}')\n\
            else:\n\
                logger.info(f'Cache not found for {model_name}, will download on first use')\n\
                \n\
    except Exception as e:\n\
        logger.error(f'Model preloading error: {e}')\n\
\n\
if __name__ == '__main__':\n\
    preload_models()\n\
    logger.info('🎉 Model pre-loading completed')\n\
"\n\
\n\
echo "🚀 Starting RAPTOR server..."\n\
exec python generic-qa-server.py' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Health check with model loading awareness
HEALTHCHECK --interval=60s --timeout=30s --start-period=600s --retries=10 \
    CMD python -c "\
import requests, sys, json; \
try: \
    response = requests.get('http://localhost:8000/health', timeout=30); \
    data = response.json(); \
    models_loaded = data.get('models', {}).get('models_loaded', False); \
    status = data.get('status', 'unhealthy'); \
    print(f'Health: {status}, Models: {models_loaded}'); \
    sys.exit(0 if models_loaded and status in ['healthy', 'degraded'] else 1) \
except Exception as e: \
    print(f'Health check failed: {e}'); \
    sys.exit(1)"

# Expose port
EXPOSE 8000

# Use entrypoint script for better startup control
ENTRYPOINT ["/app/entrypoint.sh"]