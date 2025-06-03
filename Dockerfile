FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    uvicorn[standard]==0.27.0 \
    psutil==5.9.7 \
    prometheus_client==0.19.0 \
    redis[hiredis]==5.0.1

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash raptor && \
    chown -R raptor:raptor /app && \
    mkdir -p logs && \
    chown raptor:raptor logs

USER raptor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run application
CMD ["python", "generic-qa-server.py"]