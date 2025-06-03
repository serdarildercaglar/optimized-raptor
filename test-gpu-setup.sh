#!/bin/bash

echo "🚀 GPU Setup Test for RAPTOR"
echo "============================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

# 1. Check NVIDIA GPU
log_test "Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
    echo ""
    log_info "✅ NVIDIA GPU detected"
else
    log_error "❌ nvidia-smi not found - No NVIDIA GPU or drivers not installed"
    exit 1
fi

# 2. Check Docker GPU support
log_test "Checking Docker GPU support..."

# Try different CUDA images in order of preference
CUDA_IMAGES=(
    "nvidia/cuda:latest"
    "nvidia/cuda:11.8-runtime-ubuntu22.04"
    "nvidia/cuda:12.1-runtime-ubuntu22.04" 
    "nvidia/cuda:11.8-devel-ubuntu22.04"
    "nvidia/cuda:11.2-runtime-ubuntu20.04"
)

DOCKER_GPU_OK=false
for image in "${CUDA_IMAGES[@]}"; do
    log_info "Trying CUDA image: $image"
    if docker run --rm --gpus all $image nvidia-smi &> /dev/null; then
        log_info "✅ Docker can access GPU with image: $image"
        DOCKER_GPU_OK=true
        break
    fi
done

if [ "$DOCKER_GPU_OK" = false ]; then
    log_error "❌ Docker cannot access GPU with any CUDA image"
    echo "Available options:"
    echo "1. Check nvidia-container-toolkit installation"
    echo "2. Restart Docker: sudo systemctl restart docker"
    echo "3. Configure runtime: sudo nvidia-ctk runtime configure --runtime=docker"
    exit 1
fi

# 3. Check HuggingFace cache
log_test "Checking HuggingFace cache..."
HF_CACHE_DIR="/home/$(whoami)/.cache/huggingface/hub"
if [ -d "$HF_CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1)
    MODEL_COUNT=$(find "$HF_CACHE_DIR" -name "config.json" | wc -l)
    log_info "✅ HuggingFace cache found"
    log_info "   Location: $HF_CACHE_DIR"
    log_info "   Size: $CACHE_SIZE"
    log_info "   Models: $MODEL_COUNT"
    
    # List some models
    echo ""
    echo "📋 Cached models (showing first 5):"
    find "$HF_CACHE_DIR" -name "config.json" -exec dirname {} \; | head -5 | while read dir; do
        model_name=$(basename "$(dirname "$dir")")
        echo "   - $model_name"
    done
else
    log_warn "⚠️ HuggingFace cache not found at $HF_CACHE_DIR"
    log_warn "Models will be downloaded during first run"
fi

# 4. Test CUDA with Python
log_test "Testing CUDA with Python..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available in PyTorch')
    exit(1)
"

if [ $? -eq 0 ]; then
    log_info "✅ CUDA working with PyTorch"
else
    log_error "❌ CUDA not working with PyTorch"
    exit 1
fi

# 5. Test embedding model loading
log_test "Testing HuggingFace model loading..."
python -c "
import os
import torch
from sentence_transformers import SentenceTransformer

# Set cache directory
os.environ['HF_HOME'] = '/home/$(whoami)/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/home/$(whoami)/.cache/huggingface/hub'

print('Loading multilingual-e5-large model...')
try:
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    # Check if model is on GPU
    device = next(model.parameters()).device
    print(f'Model device: {device}')
    
    # Test encoding
    text = 'Bu bir test cümlesidir.'
    embedding = model.encode(text)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Embedding type: {type(embedding)}')
    
    print('✅ Model loading and encoding successful')
    
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    log_info "✅ HuggingFace model loading successful"
else
    log_error "❌ HuggingFace model loading failed"
    exit 1
fi

# 6. Summary
echo ""
echo "🎯 GPU Setup Summary"
echo "===================="
log_info "✅ NVIDIA GPU: Available"
log_info "✅ Docker GPU: Working"
log_info "✅ HuggingFace Cache: Available"
log_info "✅ CUDA PyTorch: Working"
log_info "✅ Model Loading: Working"

echo ""
echo "🚀 Ready for GPU-accelerated RAPTOR deployment!"
echo ""
echo "Next steps:"
echo "  1. ./scripts/deploy.sh"
echo "  2. Monitor GPU usage: watch nvidia-smi"
echo "  3. Check container logs: docker logs raptor-app"
echo ""

# GPU monitoring tip
echo "💡 Pro tip: Monitor GPU usage during deployment:"
echo "   watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'"
echo ""

echo "✅ GPU test completed successfully!"