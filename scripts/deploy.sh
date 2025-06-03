#!/bin/bash

# DOSYA: scripts/deploy.sh
# AÇIKLAMA: Production deployment script for RAPTOR with GPU support

set -e

echo "🚀 RAPTOR Production Deployment (GPU Optimized)"
echo "================================================"

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

# Check Docker Compose availability
check_docker_compose() {
    if detect_docker_compose; then
        return 0
    else
        return 1
    fi
}

# Check prerequisites including GPU
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        echo "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! check_docker_compose; then
        log_error "Docker Compose is required but not installed"
        echo "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    log_info "Using Docker Compose command: $DOCKER_COMPOSE_CMD"
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
    fi
    
    # Check GPU support
    log_info "Checking GPU support..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        log_info "✅ GPU detected: $GPU_INFO"
        
        # Check Docker GPU support
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_info "✅ Docker GPU support verified"
        else
            log_warn "⚠️ Docker GPU support not available"
            log_warn "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        log_warn "⚠️ No GPU detected (nvidia-smi not found)"
        log_warn "RAPTOR will run on CPU (slower performance)"
    fi
    
    # Check HuggingFace cache
    log_info "Checking HuggingFace cache..."
    HF_CACHE_DIR="/home/$(whoami)/.cache/huggingface"
    if [ -d "$HF_CACHE_DIR" ]; then
        CACHE_SIZE=$(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
        log_info "✅ HuggingFace cache found: $CACHE_SIZE"
    else
        log_warn "⚠️ HuggingFace cache not found at $HF_CACHE_DIR"
        log_warn "Models will be downloaded during startup (slower)"
    fi
    
    # Check .env file
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
    
    # Check OPENAI_API_KEY
    source .env
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        log_error "OPENAI_API_KEY not set in .env file"
        exit 1
    fi
    
    # Check RAPTOR tree
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

# Create missing configuration files
create_configs() {
    log_step "Creating configuration files..."
    
    # Create Redis config if missing
    if [ ! -f "config/redis.conf" ]; then
        log_info "Creating Redis configuration..."
        cat > config/redis.conf << 'EOF'
# Redis Production Configuration
bind 127.0.0.1
port 6379
timeout 300
tcp-keepalive 60

# Memory
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename raptor.rdb

# Logging
loglevel notice
logfile ""

# Performance
databases 1
hz 10
EOF
    fi
    
    # Create Prometheus config if missing
    if [ ! -f "monitoring/prometheus/prometheus.yml" ]; then
        log_info "Creating Prometheus configuration..."
        mkdir -p monitoring/prometheus
        cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'raptor-app'
    static_configs:
      - targets: ['raptor-app:8000']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 10s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    scrape_interval: 10s
EOF
    fi
    
    log_info "Configuration files created ✅"
}

# Build images
build_images() {
    log_step "Building GPU-optimized Docker images..."
    
    # Clean up old images if requested
    if [ "${CLEAN_BUILD:-false}" = "true" ]; then
        log_info "Cleaning old images..."
        $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --rmi all --remove-orphans || true
    fi
    
    # Build images
    log_info "Building images with GPU support..."
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE build --no-cache
    
    if [ $? -eq 0 ]; then
        log_info "Docker build completed successfully ✅"
    else
        log_error "Docker build failed ❌"
        exit 1
    fi
}

# Deploy services
deploy_services() {
    log_step "Deploying services..."
    
    # Stop existing services
    log_info "Stopping existing services..."
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans || true
    
    # Start services
    log_info "Starting services with GPU support..."
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE up -d
    
    if [ $? -eq 0 ]; then
        log_info "Services deployed successfully ✅"
    else
        log_error "Service deployment failed ❌"
        exit 1
    fi
}

# Enhanced health check with GPU startup time
health_check() {
    log_step "Performing health checks..."
    
    # Wait for services to start (increased for GPU model loading)
    log_info "Waiting for services to start (120 seconds for GPU model loading)..."
    sleep 120
    
    local health_ok=true
    
    # Check Redis
    log_info "Checking Redis..."
    if $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE exec -T redis redis-cli ping | grep -q "PONG"; then
        log_info "✅ Redis is healthy"
    else
        log_error "❌ Redis health check failed"
        health_ok=false
    fi
    
    # Check RAPTOR app (more attempts for GPU startup)
    log_info "Checking RAPTOR app (GPU model loading may take time)..."
    local app_healthy=false
    for i in {1..10}; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_info "✅ RAPTOR app is healthy"
            app_healthy=true
            break
        else
            log_warn "RAPTOR app check attempt $i/10 failed, retrying in 15s..."
            sleep 15
        fi
    done
    
    if [ "$app_healthy" = false ]; then
        log_error "❌ RAPTOR app health check failed"
        log_info "Checking container logs for debugging..."
        $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs --tail=50 raptor-app
        health_ok=false
    fi
    
    # Check Prometheus
    log_info "Checking Prometheus..."
    if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        log_info "✅ Prometheus is healthy"
    else
        log_warn "⚠️ Prometheus health check failed (non-critical)"
    fi
    
    # Check Grafana
    log_info "Checking Grafana..."
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        log_info "✅ Grafana is healthy"
    else
        log_warn "⚠️ Grafana health check failed (non-critical)"
    fi
    
    if [ "$health_ok" = true ]; then
        log_info "Health checks completed ✅"
        return 0
    else
        log_error "Critical health checks failed ❌"
        return 1
    fi
}

# Show deployment info with GPU details
show_info() {
    echo ""
    echo "🎉 GPU-Optimized RAPTOR Deployment Successful!"
    echo "=============================================="
    echo ""
    echo "📋 Service URLs:"
    echo "   🌐 RAPTOR API:     http://localhost:8000"
    echo "   🔍 Health Check:   http://localhost:8000/health"
    echo "   📊 Prometheus:     http://localhost:9090"
    echo "   📈 Grafana:        http://localhost:3000 (admin/admin)"
    echo ""
    echo "🚀 GPU Information:"
    if command -v nvidia-smi &> /dev/null; then
        echo "   $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | head -1)"
        echo "   📊 Monitor GPU: watch nvidia-smi"
    else
        echo "   ⚠️ Running on CPU (no GPU detected)"
    fi
    echo ""
    echo "📋 Useful Commands:"
    echo "   📜 View logs:       $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE logs -f raptor-app"
    echo "   🔧 Check GPU:       docker exec raptor-app nvidia-smi"
    echo "   ⏹️  Stop services:   $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down"
    echo "   🔄 Restart:         $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE restart raptor-app"
    echo "   📊 Service status:  $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE ps"
    echo ""
    echo "🧪 Quick Test:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "📁 Cache Locations:"
    echo "   🧠 HuggingFace:     /home/$(whoami)/.cache/huggingface (mounted)"
    echo "   💾 Embeddings:      ./embedding_cache/"
    echo "   🔍 Queries:         ./query_cache/"
    echo ""
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --remove-orphans
    log_info "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    $DOCKER_COMPOSE_CMD -f $COMPOSE_FILE down --volumes --remove-orphans
    docker system prune -f
    log_info "Cleanup completed"
}

# Show help
show_help() {
    echo "🚀 RAPTOR Production Deployment Script (GPU Optimized)"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -e, --env ENV       Environment (development|staging|production) [default: production]"
    echo "  -c, --clean         Clean build (remove old images)"
    echo "  -r, --rollback      Rollback deployment"
    echo "  --cleanup           Full cleanup (remove all containers and volumes)"
    echo "  --no-build          Skip building images"
    echo "  --no-health         Skip health checks"
    echo "  --gpu-test          Test GPU setup only"
    echo "  --dry-run           Show what would be done without executing"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT         Deployment environment"
    echo "  CLEAN_BUILD         Clean build flag (true/false)"
    echo "  SKIP_BUILD          Skip build flag (true/false)"
    echo "  SKIP_HEALTH         Skip health checks (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Standard GPU production deployment"
    echo "  $0 --env development                  # Development deployment"
    echo "  $0 --clean                           # Clean build deployment"
    echo "  $0 --gpu-test                        # Test GPU setup only"
    echo "  $0 --rollback                        # Rollback current deployment"
    echo ""
}

# GPU test function
test_gpu_only() {
    log_step "Testing GPU setup..."
    
    if [ -f "test-gpu-setup.sh" ]; then
        chmod +x test-gpu-setup.sh
        ./test-gpu-setup.sh
    else
        log_error "test-gpu-setup.sh not found"
        exit 1
    fi
}

# Parse command line arguments
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
            -r|--rollback)
                rollback
                exit 0
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            --no-build)
                SKIP_BUILD=true
                shift
                ;;
            --no-health)
                SKIP_HEALTH=true
                shift
                ;;
            --gpu-test)
                test_gpu_only
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main deployment function
main() {
    echo "🚀 RAPTOR Production Deployment (GPU Optimized)"
    echo "Environment: $ENVIRONMENT"
    echo "================================================"
    
    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "🧪 DRY RUN MODE - No changes will be made"
        echo "Would execute:"
        echo "1. Check prerequisites (including GPU)"
        echo "2. Create configuration files"
        echo "3. Build GPU-optimized Docker images"
        echo "4. Deploy services with GPU support"
        echo "5. Run health checks (with extended GPU startup time)"
        exit 0
    fi
    
    # Execute deployment steps
    check_prerequisites
    create_configs
    
    if [ "${SKIP_BUILD:-false}" != "true" ]; then
        build_images
    else
        log_info "Skipping image build"
    fi
    
    deploy_services
    
    if [ "${SKIP_HEALTH:-false}" != "true" ]; then
        if health_check; then
            show_info
        else
            log_error "Health checks failed, rolling back..."
            rollback
            exit 1
        fi
    else
        log_info "Skipping health checks"
        show_info
    fi
}

# Handle signals for graceful shutdown
trap rollback SIGINT SIGTERM

# Parse arguments and run main
parse_args "$@"
main

echo "✅ GPU-optimized RAPTOR deployment completed successfully!"