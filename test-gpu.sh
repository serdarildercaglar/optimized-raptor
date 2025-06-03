#!/bin/bash

# DOSYA: test-gpu.sh
# AÇIKLAMA: Simple GPU test for deployment script

echo "🧪 Quick GPU Test"
echo "================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
        log_info "✅ GPU detected: $GPU_INFO"
    else
        log_error "❌ nvidia-smi failed to get GPU info"
        exit 1
    fi
else
    log_error "❌ nvidia-smi not found"
    exit 1
fi

# 2. Check nvidia-container-toolkit
if command -v nvidia-ctk &> /dev/null; then
    log_info "✅ nvidia-container-toolkit found"
else
    log_warn "⚠️ nvidia-container-toolkit not found"
    log_warn "Install with: sudo apt install nvidia-container-toolkit"
    exit 1
fi

# 3. Check Docker daemon GPU support
if docker info 2>/dev/null | grep -q "nvidia"; then
    log_info "✅ Docker daemon has nvidia runtime"
else
    log_warn "⚠️ Docker daemon nvidia runtime not configured"
    log_warn "Configure with: sudo nvidia-ctk runtime configure --runtime=docker"
    log_warn "Then restart: sudo systemctl restart docker"
fi

# 4. Test simple GPU access with correct image
log_info "Testing GPU access..."
log_info "Pulling correct CUDA image..."

# Pull the correct image first
if docker pull nvidia/cuda:11.8.0-base-ubuntu22.04 > /dev/null 2>&1; then
    log_info "✅ CUDA image pulled successfully"
else
    log_error "❌ Failed to pull CUDA image"
    exit 1
fi

# Test GPU access
if docker run --rm --gpus all --entrypoint nvidia-smi nvidia/cuda:11.8.0-base-ubuntu22.04 --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "GeForce\|Quadro\|Tesla\|RTX\|GTX"; then
    log_info "✅ Docker GPU access working with nvidia/cuda:11.8.0-base-ubuntu22.04"
    echo "🎉 GPU test passed!"
    exit 0
else
    log_error "❌ Docker GPU access failed"
    echo ""
    echo "🔧 Quick fix commands:"
    echo "sudo apt update"
    echo "sudo apt install nvidia-container-toolkit"
    echo "sudo nvidia-ctk runtime configure --runtime=docker"
    echo "sudo systemctl restart docker"
    echo ""
    echo "🧪 Manual test:"
    echo "docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
    echo ""
    exit 1
fi