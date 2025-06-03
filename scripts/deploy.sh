#!/bin/bash

# DOSYA: scripts/deploy.sh
# AÇIKLAMA: Production deployment script for RAPTOR with GPU support and model pre-loading

set -e

echo "🚀 RAPTOR Production Deployment (GPU + Model Pre-loading)"
echo "========================================================="

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
BUILD_TARGET=${BUILD_TARGET:-production}
COMPOSE_FILE="docker-compose.production.yml"
PROJECT_NAME="raptor"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

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

log_model() {
    echo -e "${PURPLE}[MODEL]${NC} $1"
}

# Detect Docker Compose command (v1 vs v2)
detect_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
        return 0
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
        return 0
    else
        return 1
    fi
}

# Enhanced GPU check with better error handling
check_gpu_setup() {
    log_step "Checking GPU setup..."
    
    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ $? -eq 0 ] && [ -n "$GPU_INFO" ]; then
            log_info "✅ GPU detected: $GPU_INFO"
        else
            log_error "❌ nvidia-smi failed to get GPU info"
            log_warn "RAPTOR will run on CPU (slower performance)"
            GPU_RUNTIME_OK=false
            return 0
        fi
        
        # Check nvidia-container-toolkit
        if command -v nvidia-ctk &> /dev/null; then
            log_info "✅ nvidia-container-toolkit found"
        else
            log_error "❌ nvidia-container-toolkit not found"
            log_error "Please install:"
            log_error "  sudo apt update"
            log_error "  sudo apt install nvidia-container-toolkit"
            log_error "  sudo nvidia-ctk runtime configure --runtime=docker"
            log_error "  sudo systemctl restart docker"
            return 1
        fi
        
        # Check Docker daemon configuration
        if docker info 2>/dev/null | grep -q "nvidia"; then
            log_info "✅ Docker nvidia runtime configured"
        else
            log_warn "⚠️ Docker nvidia runtime not fully configured"
            log_warn "Run: sudo nvidia-ctk runtime configure --runtime=docker"
            log_warn "Then: sudo systemctl restart docker"
        fi
        
        # Test Docker GPU access with simpler approach
        log_info "Testing Docker GPU access..."
        
        # Try multiple CUDA images in order of preference
        CUDA_IMAGES=(
            "nvidia/cuda:11.8.0-base-ubuntu22.04"
            "nvidia/cuda:12.1.0-base-ubuntu22.04"
            "nvidia/cuda:11.2.0-base-ubuntu20.04"
        )
        
        GPU_RUNTIME_OK=false
        for image in "${CUDA_IMAGES[@]}"; do
            log_info "Testing CUDA image: $image"
            
            # Pull image first to avoid network issues during test
            if docker pull "$image" > /dev/null 2>&1; then
                log_info "✅ Successfully pulled $image"
                
                # Test with --gpus all first
                if timeout 30s docker run --rm --gpus all --entrypoint nvidia-smi "$image" --version &> /dev/null; then
                    log_info "✅ Docker GPU support verified with $image (--gpus all)"
                    GPU_RUNTIME_OK=true
                    VERIFIED_CUDA_IMAGE="$image"
                    break
                fi
                
                # Try with --runtime=nvidia as fallback
                if timeout 30s docker run --rm --runtime=nvidia --entrypoint nvidia-smi "$image" --version &> /dev/null; then
                    log_info "✅ Docker GPU support verified with $image (--runtime=nvidia)"
                    GPU_RUNTIME_OK=true
                    VERIFIED_CUDA_IMAGE="$image"
                    break
                fi
            else
                log_warn "⚠️ Failed to pull $image"
            fi
        done
        
        if [ "$GPU_RUNTIME_OK" = false ]; then
            log_error "❌ Docker GPU support test failed with all CUDA images"
            log_error ""
            log_error "🔧 Troubleshooting steps:"
            log_error "1. Check if nvidia-container-toolkit is installed:"
            log_error "   dpkg -l | grep nvidia-container-toolkit"
            log_error ""
            log_error "2. Install nvidia-container-toolkit:"
            log_error "   sudo apt update"
            log_error "   sudo apt install nvidia-container-toolkit"
            log_error ""
            log_error "3. Configure Docker runtime:"
            log_error "   sudo nvidia-ctk runtime configure --runtime=docker"
            log_error "   sudo systemctl restart docker"
            log_error ""
            log_error "4. Test manually:"
            log_error "   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi"
            log_error ""
            
            # Ask user if they want to continue without GPU
            if [ "${FORCE_CPU:-false}" != "true" ]; then
                echo -e "${YELLOW}Do you want to continue with CPU-only deployment? (y/N):${NC}"
                read -r response
                if [[ ! "$response" =~ ^[Yy]$ ]]; then
                    log_error "Deployment cancelled. Fix GPU support and try again."
                    exit 1
                fi
                log_warn "Continuing with CPU-only deployment..."
                GPU_RUNTIME_OK=false
            fi
        fi
    else
        log_warn "⚠️ No GPU detected (nvidia-smi not found)"
        log_warn "RAPTOR will run on CPU (slower performance)"
        GPU_RUNTIME_OK=false
    fi
    
    return 0
}

# Enhanced HuggingFace cache check and setup
setup_huggingface_cache() {
    log_step "Setting up HuggingFace cache..."
    
    # Get current user's cache directory
    HF_CACHE_DIR="${HOME}/.cache/huggingface"
    HF_HUB_DIR="${HF_CACHE_DIR}/hub"
    
    log_info "HuggingFace cache directory: $HF_CACHE_DIR"
    
    # Create cache directories
    mkdir -p "$HF_HUB_DIR"
    mkdir -p "${HF_CACHE_DIR}/datasets"
    
    if [ -d "$HF_HUB_DIR" ]; then
        CACHE_SIZE=$(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
        MODEL_COUNT=$(find "$HF_HUB_DIR" -name "config.json" 2>/dev/null | wc -l)
        log_info "✅ HuggingFace cache found: $CACHE_SIZE ($MODEL_COUNT models)"
        
        # List some important models for RAPTOR
        REQUIRED_MODELS=(
            "intfloat/multilingual-e5-large"
            "sentence-transformers/multi-qa-mpnet-base-cos-v1"
        )
        
        log_info "Checking required models..."
        for model in "${REQUIRED_MODELS[@]}"; do
            model_dir=$(echo "$model" | tr '/' '--')
            if find "$HF_HUB_DIR" -name "*${model_dir}*" -type d | grep -q .; then
                log_info "  ✅ $model (cached)"
            else
                log_warn "  ⚠️ $model (will be downloaded)"
            fi
        done
    else
        log_warn "⚠️ HuggingFace cache directory not found"
        log_warn "Models will be downloaded during first startup (may take 10-15 minutes)"
    fi
    
    # Export cache path for docker-compose
    export HF_CACHE_PATH="$HF_CACHE_DIR"
}

# Model pre-downloading function
pre_download_models() {
    log_step "Pre-downloading required models..."
    
    if [ "${SKIP_MODEL_DOWNLOAD:-false}" = "true" ]; then
        log_info "Skipping model download (SKIP_MODEL_DOWNLOAD=true)"
        return 0
    fi
    
    log_model "Starting model pre-download..."
    
    # Create a temporary container for model downloading
    cat > download_models.py << 'EOF'
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

print("🚀 Pre-downloading RAPTOR models...")

# Set cache directories
os.environ['HF_HOME'] = '/cache'
os.environ['TRANSFORMERS_CACHE'] = '/cache/hub'

models_to_download = [
    'intfloat/multilingual-e5-large',
    'sentence-transformers/multi-qa-mpnet-base-cos-v1'
]

for model_name in models_to_download:
    try:
        print(f"📥 Downloading {model_name}...")
        
        if 'sentence-transformers' in model_name:
            model = SentenceTransformer(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        
        print(f"✅ {model_name} downloaded successfully")
        del model
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

print("🎉 Model pre-download completed!")
EOF
    
    # Run model download in Docker container with verified image
    log_model "Running model download container..."
    
    # Use verified CUDA image or fallback
    DOWNLOAD_IMAGE="${VERIFIED_CUDA_IMAGE:-nvidia/cuda:11.8.0-base-ubuntu22.04}"
    
    if docker run --rm \
        --gpus all \
        -v "${HOME}/.cache/huggingface:/cache" \
        -v "$(pwd)/download_models.py:/download_models.py" \
        "$DOWNLOAD_IMAGE" \
        bash -c "
            apt-get update && apt-get install -y python3 python3-pip && \
            pip install torch sentence-transformers transformers accelerate && \
            python3 /download_models.py
        "; then
        log_model "✅ Model pre-download completed successfully"
        rm -f download_models.py
    else
        log_warn "⚠️ Model pre-download failed, models will be downloaded during startup"
        rm -f download_models.py
    fi
}

# Enhanced health check with model loading awareness
enhanced_health_check() {
    log_step "Performing enhanced health checks..."
    
    # Wait for initial startup
    log_info "Waiting for initial container startup (60 seconds)..."
    sleep 60
    
    local health_ok=true
    local max_wait_time=900  # 15 minutes for model loading
    local check_interval=30  # Check every 30 seconds
    local elapsed_time=0
    
    # Check Redis first
    log_info "Checking Redis..."
    if $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE exec -T redis redis-cli ping | grep -q "PONG"; then
        log_info "✅ Redis is healthy"
    else
        log_error "❌ Redis health check failed"
        health_ok=false
    fi
    
    # Enhanced RAPTOR app check with model loading awareness
    log_info "Checking RAPTOR app with model loading monitoring..."
    log_model "This may take 5-15 minutes for model loading on first run..."
    
    local app_healthy=false
    local last_log_time=0
    
    while [ $elapsed_time -lt $max_wait_time ]; do
        # Check basic connectivity
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            # Get detailed health status
            health_response=$(curl -s http://localhost:8000/health || echo '{}')
            models_loaded=$(echo "$health_response" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('models_loaded', False))
except:
    print(False)
" 2>/dev/null || echo "False")
            
            if [ "$models_loaded" = "True" ]; then
                log_info "✅ RAPTOR app is healthy and models are loaded"
                app_healthy=true
                break
            else
                # Show progress every 60 seconds
                if [ $((elapsed_time - last_log_time)) -ge 60 ]; then
                    log_model "⏳ Models still loading... (${elapsed_time}s elapsed)"
                    
                    # Show container logs for model loading progress
                    echo "📋 Recent container logs:"
                    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs --tail=5 raptor-app | grep -E "(Loading|Downloading|Model|✅|❌)" || true
                    echo ""
                    
                    last_log_time=$elapsed_time
                fi
            fi
        else
            if [ $((elapsed_time - last_log_time)) -ge 60 ]; then
                log_warn "⏳ RAPTOR app not responding yet... (${elapsed_time}s elapsed)"
                last_log_time=$elapsed_time
            fi
        fi
        
        sleep $check_interval
        elapsed_time=$((elapsed_time + check_interval))
    done
    
    if [ "$app_healthy" = false ]; then
        log_error "❌ RAPTOR app health check failed after ${max_wait_time}s"
        log_error "Showing container logs for debugging:"
        $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50 raptor-app
        health_ok=false
    fi
    
    # Check other services (non-critical)
    log_info "Checking monitoring services..."
    
    if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_info "✅ Prometheus is healthy"
    else
        log_warn "⚠️ Prometheus health check failed (non-critical)"
    fi
    
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        log_info "✅ Grafana is healthy"
    else
        log_warn "⚠️ Grafana health check failed (non-critical)"
    fi
    
    if [ "$health_ok" = true ]; then
        log_info "✅ Enhanced health checks completed successfully"
        return 0
    else
        log_error "❌ Critical health checks failed"
        return 1
    fi
}

# Enhanced deployment info with GPU and model details
show_enhanced_info() {
    echo ""
    echo "🎉 RAPTOR Deployment Successful ($DEPLOYMENT_MODE Mode)!"
    echo "======================================================="
    echo ""
    echo "📋 Service URLs:"
    echo "   🌐 RAPTOR API:     http://localhost:8000"
    echo "   🔍 Health Check:   http://localhost:8000/health"
    echo "   📊 Prometheus:     http://localhost:9090"
    echo "   📈 Grafana:        http://localhost:3000 (admin/admin)"
    echo ""
    
    if [ "$DEPLOYMENT_MODE" = "GPU" ] && command -v nvidia-smi &> /dev/null; then
        echo "🚀 GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read line; do
            echo "   GPU: $line"
        done
        echo "   📊 Monitor GPU: watch nvidia-smi"
        echo "   🔧 Check GPU in container: docker exec raptor-app nvidia-smi"
    else
        echo "🚀 System Information:"
        if [ "$DEPLOYMENT_MODE" = "CPU" ]; then
            echo "   💻 Running in CPU-only mode"
            echo "   ⚡ Performance: Slower than GPU but still functional"
        else
            echo "   ⚠️ GPU mode selected but GPU monitoring unavailable"
        fi
    fi
    
    echo ""
    echo "🧠 Model Information:"
    echo "   📁 Cache: ${HOME}/.cache/huggingface"
    if [ -d "${HOME}/.cache/huggingface/hub" ]; then
        model_count=$(find "${HOME}/.cache/huggingface/hub" -name "config.json" 2>/dev/null | wc -l)
        cache_size=$(du -sh "${HOME}/.cache/huggingface" 2>/dev/null | cut -f1 || echo "unknown")
        echo "   📊 Models: $model_count ($cache_size)"
    fi
    echo "   🔧 Mode: $DEPLOYMENT_MODE"
    
    echo ""
    echo "📋 Useful Commands:"
    echo "   📜 View logs:       $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs -f raptor-app"
    if [ "$DEPLOYMENT_MODE" = "GPU" ]; then
        echo "   🔧 Check GPU:       docker exec raptor-app nvidia-smi"
    else
        echo "   🔧 Check CPU:       docker exec raptor-app htop"
    fi
    echo "   🧠 Model status:    curl http://localhost:8000/health | jq '.models.models_loaded'"
    echo "   ⏹️  Stop services:   $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down"
    echo "   🔄 Restart:         $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE restart raptor-app"
    echo "   📊 Service status:  $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE ps"
    echo ""
    echo "🧪 Quick Test:"
    echo "   curl http://localhost:8000/health"
    echo ""
    
    if [ "$DEPLOYMENT_MODE" = "CPU" ]; then
        echo "💡 Performance Tips for CPU Mode:"
        echo "   - First startup will be slower (15-20 minutes)"
        echo "   - Response times will be 3-5x slower than GPU"
        echo "   - Consider upgrading to GPU for production use"
        echo ""
    fi
}

# Enhanced prerequisite check
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Docker check
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        echo "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Docker Compose check
    if ! detect_docker_compose; then
        log_error "Docker Compose is required but not installed"
        echo "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    log_info "Using Docker Compose command: $DOCKER_COMPOSE_CMD"
    
    # Docker running check
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
    fi
    
    # GPU setup check
    check_gpu_setup
    
    # HuggingFace cache setup
    setup_huggingface_cache
    
    # Environment file check
    if [ ! -f ".env" ]; then
        log_warn ".env file not found, creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_warn "Please edit .env file with your configuration"
            log_warn "Required: OPENAI_API_KEY"
            exit 1
        else
            log_error ".env.example file not found"
            exit 1
        fi
    fi
    
    # API key check
    source .env
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        log_error "OPENAI_API_KEY not set in .env file"
        exit 1
    fi
    
    # RAPTOR tree check
    if [ ! -d "vectordb" ] || [ ! -f "vectordb/raptor-production" ]; then
        log_error "RAPTOR tree not found at vectordb/raptor-production"
        log_error "Please build RAPTOR tree first:"
        log_error "  python build-raptor-production.py data.txt"
        exit 1
    fi
    
    # Create required directories
    mkdir -p logs config monitoring/prometheus monitoring/grafana nginx ssl embedding_cache query_cache
    
    log_info "Prerequisites check passed ✅"
}

# Main deployment with GPU/CPU fallback
main() {
    echo "🚀 RAPTOR Production Deployment (GPU + Model Pre-loading)"
    echo "Environment: $ENVIRONMENT"
    echo "========================================================="
    
    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "🧪 DRY RUN MODE - No changes will be made"
        echo "Would execute:"
        echo "1. Check prerequisites (including GPU)"
        echo "2. Setup HuggingFace cache"
        echo "3. Pre-download models (optional)"
        echo "4. Build GPU-optimized Docker images"
        echo "5. Deploy services with GPU support"
        echo "6. Run enhanced health checks (with model loading monitoring)"
        exit 0
    fi
    
    # Execute deployment steps
    check_prerequisites
    
    # Determine deployment mode based on GPU availability
    if [ "$GPU_RUNTIME_OK" = true ]; then
        COMPOSE_FILE="docker-compose.production.yml"
        DEPLOYMENT_MODE="GPU"
        log_info "🚀 GPU deployment mode selected"
        
        # Model pre-download for GPU (faster with GPU)
        if [ "${PRE_DOWNLOAD_MODELS:-true}" = "true" ]; then
            pre_download_models
        fi
    else
        COMPOSE_FILE="docker-compose.cpu.yml"
        DEPLOYMENT_MODE="CPU"
        log_info "💻 CPU-only deployment mode selected"
        
        # Skip model pre-download for CPU (will be slower anyway)
        log_info "Skipping model pre-download for CPU deployment"
    fi
    
    # Build and deploy
    if [ "${SKIP_BUILD:-false}" != "true" ]; then
        log_step "Building $DEPLOYMENT_MODE-optimized Docker images..."
        if [ "${CLEAN_BUILD:-false}" = "true" ]; then
            log_info "Cleaning old images..."
            $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --rmi all --remove-orphans || true
        fi
        
        $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE build --no-cache
        
        if [ $? -eq 0 ]; then
            log_info "Docker build completed successfully ✅"
        else
            log_error "Docker build failed ❌"
            
            # If GPU build failed, try CPU fallback
            if [ "$DEPLOYMENT_MODE" = "GPU" ] && [ "${AUTO_FALLBACK_CPU:-true}" = "true" ]; then
                log_warn "GPU build failed, trying CPU fallback..."
                COMPOSE_FILE="docker-compose.cpu.yml"
                DEPLOYMENT_MODE="CPU"
                
                $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE build --no-cache
                if [ $? -eq 0 ]; then
                    log_info "CPU fallback build completed successfully ✅"
                else
                    log_error "CPU fallback build also failed ❌"
                    exit 1
                fi
            else
                exit 1
            fi
        fi
    else
        log_info "Skipping image build"
    fi
    
    # Deploy services
    log_step "Deploying services with $DEPLOYMENT_MODE support..."
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans || true
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE up -d
    
    if [ $? -eq 0 ]; then
        log_info "Services deployed successfully ✅"
    else
        log_error "Service deployment failed ❌"
        exit 1
    fi
    
    # Enhanced health check
    if [ "${SKIP_HEALTH:-false}" != "true" ]; then
        if enhanced_health_check; then
            show_enhanced_info
        else
            log_error "Health checks failed, rolling back..."
            $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans
            exit 1
        fi
    else
        log_info "Skipping health checks"
        show_enhanced_info
    fi
}

# Enhanced argument parsing
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -c|--clean)
                CLEAN_BUILD=true
                shift
                ;;
            --no-build)
                SKIP_BUILD=true
                shift
                ;;
            --no-health)
                SKIP_HEALTH=true
                shift
                ;;
            --no-models)
                PRE_DOWNLOAD_MODELS=false
                shift
                ;;
            --skip-model-download)
                SKIP_MODEL_DOWNLOAD=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -r|--rollback)
                log_warn "Rolling back deployment..."
                $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans
                exit 0
                ;;
            --cleanup)
                log_info "Cleaning up..."
                $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --volumes --remove-orphans
                docker system prune -f
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Handle signals for graceful shutdown
trap '$DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans' SIGINT SIGTERM

# Parse arguments and run main
parse_args "$@"
main

echo "✅ GPU-optimized RAPTOR deployment with model pre-loading completed successfully!"