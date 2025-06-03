#!/bin/bash

# DOSYA: quick-setup.sh
# AÇIKLAMA: One-click RAPTOR setup with GPU support

set -e

echo "🚀 RAPTOR Quick Setup with GPU Support"
echo "======================================"

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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 1. Check if running as root
if [ "$EUID" -eq 0 ]; then
    log_error "Please don't run this script as root"
    exit 1
fi

# 2. Check Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    log_error "This script is for Ubuntu/Debian systems only"
    exit 1
fi

# 3. Install nvidia-container-toolkit
log_step "Installing nvidia-container-toolkit..."

# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt update
sudo apt install -y nvidia-container-toolkit

log_info "✅ nvidia-container-toolkit installed"

# 4. Configure Docker runtime
log_step "Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

log_info "✅ Docker runtime configured"

# 5. Test GPU access
log_step "Testing GPU access..."

# Wait for Docker to restart
sleep 5

# Pull correct CUDA image
log_info "Pulling CUDA image..."
docker pull nvidia/cuda:11.8.0-base-ubuntu22.04

# Test GPU
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    log_info "✅ GPU test successful!"
    
    # Show GPU info
    echo ""
    echo "🚀 Your GPU Information:"
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
else
    log_error "❌ GPU test failed"
    echo ""
    echo "🔧 Manual troubleshooting:"
    echo "1. Check NVIDIA drivers: nvidia-smi"
    echo "2. Restart Docker: sudo systemctl restart docker" 
    echo "3. Test manually: docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
    exit 1
fi

# 6. Check .env file
log_step "Checking environment configuration..."

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        log_info "Creating .env from template..."
        cp .env.example .env
        log_warn "⚠️ Please edit .env file and add your OPENAI_API_KEY"
        echo ""
        echo "nano .env"
        echo ""
        echo "After editing .env, run: ./scripts/deploy.sh"
        exit 0
    else
        log_error ".env.example not found"
        exit 1
    fi
fi

# Check API key
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    log_error "OPENAI_API_KEY not set in .env file"
    echo ""
    echo "Please edit .env file:"
    echo "nano .env"
    echo ""
    echo "Then run: ./scripts/deploy.sh"
    exit 1
fi

# 7. Check RAPTOR tree
if [ ! -f "vectordb/raptor-production" ]; then
    log_warn "RAPTOR tree not found"
    
    if [ -f "data.txt" ]; then
        log_info "Building RAPTOR tree from data.txt..."
        python build-raptor-production.py data.txt
    else
        log_error "No data.txt found for building RAPTOR tree"
        echo ""
        echo "Create your document:"
        echo "echo 'Your document content here' > data.txt"
        echo ""
        echo "Then build tree:"
        echo "python build-raptor-production.py data.txt"
        echo ""
        echo "Finally deploy:"
        echo "./scripts/deploy.sh"
        exit 1
    fi
fi

# 8. All checks passed - deploy!
log_step "All prerequisites met - starting deployment..."
echo ""
echo "🚀 Running RAPTOR deployment with GPU support..."
echo ""

chmod +x scripts/deploy.sh
./scripts/deploy.sh

echo ""
echo "🎉 RAPTOR setup completed!"
echo ""
echo "🌐 Access your RAPTOR system at: http://localhost:8000"
echo "🔍 Health check: curl http://localhost:8000/health"
echo ""