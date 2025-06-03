#!/bin/bash

# DOSYA: scripts/deploy.sh
# AÇIKLAMA: Production deployment script for RAPTOR

set -e

echo "🚀 RAPTOR Production Deployment"
echo "================================"

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

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        echo "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        echo "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker service."
        exit 1
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
    if [ ! -d "vectordb" ] || [ ! -f "vectordb/raptor-optimized" ]; then
        log_error "RAPTOR tree not found at vectordb/raptor-optimized"
        log_error "Please build RAPTOR tree first:"
        log_error "  python build-raptor-production.py data.txt"
        exit 1
    fi
    
    # Create required directories
    mkdir -p logs config monitoring/prometheus monitoring/grafana nginx ssl
    
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
    
    # Create basic nginx config if missing
    if [ ! -f "nginx/nginx.conf" ]; then
        log_info "Creating Nginx configuration..."
        mkdir -p nginx
        cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream raptor_backend {
        server raptor-app:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://raptor_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /ws/ {
            proxy_pass http://raptor_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }
    }
}
EOF
    fi
    
    log_info "Configuration files created ✅"
}

# Create minimal docker-compose if missing
create_docker_compose() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_info "Creating Docker Compose configuration..."
        cat > $COMPOSE_FILE << 'EOF'
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: raptor-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - raptor-network

  raptor-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: raptor-app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - RAPTOR_ENV=production
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD:-}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8000
    volumes:
      - ./vectordb:/app/vectordb:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health', timeout=5)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - raptor-network

  prometheus:
    image: prom/prometheus:latest
    container_name: raptor-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - raptor-network

  grafana:
    image: grafana/grafana:latest
    container_name: raptor-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - raptor-network

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  raptor-network:
    driver: bridge
EOF
    fi
}

# Create minimal Dockerfile if missing
create_dockerfile() {
    if [ ! -f "Dockerfile" ]; then
        log_info "Creating Dockerfile..."
        cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir uvicorn[standard] psutil prometheus_client redis[hiredis]

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
EOF
    fi
}

# Build images
build_images() {
    log_step "Building Docker images..."
    
    # Clean up old images if requested
    if [ "${CLEAN_BUILD:-false}" = "true" ]; then
        log_info "Cleaning old images..."
        docker-compose -f $COMPOSE_FILE down --rmi all --remove-orphans || true
    fi
    
    # Build images
    log_info "Building images with build target: $BUILD_TARGET"
    docker-compose -f $COMPOSE_FILE build --no-cache
    
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
    docker-compose -f $COMPOSE_FILE down --remove-orphans || true
    
    # Start services
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    if [ $? -eq 0 ]; then
        log_info "Services deployed successfully ✅"
    else
        log_error "Service deployment failed ❌"
        exit 1
    fi
}

# Health check
health_check() {
    log_step "Performing health checks..."
    
    # Wait for services to start
    log_info "Waiting for services to start (60 seconds)..."
    sleep 60
    
    local health_ok=true
    
    # Check Redis
    log_info "Checking Redis..."
    if docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping | grep -q "PONG"; then
        log_info "✅ Redis is healthy"
    else
        log_error "❌ Redis health check failed"
        health_ok=false
    fi
    
    # Check RAPTOR app (multiple attempts)
    log_info "Checking RAPTOR app..."
    local app_healthy=false
    for i in {1..5}; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_info "✅ RAPTOR app is healthy"
            app_healthy=true
            break
        else
            log_warn "RAPTOR app check attempt $i/5 failed, retrying..."
            sleep 10
        fi
    done
    
    if [ "$app_healthy" = false ]; then
        log_error "❌ RAPTOR app health check failed"
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

# Show deployment info
show_info() {
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "====================================="
    echo ""
    echo "📋 Service URLs:"
    echo "   🌐 RAPTOR API:     http://localhost:8000"
    echo "   🔍 Health Check:   http://localhost:8000/health"
    echo "   📊 Prometheus:     http://localhost:9090"
    echo "   📈 Grafana:        http://localhost:3000 (admin/admin)"
    echo ""
    echo "📋 Useful Commands:"
    echo "   📜 View logs:       docker-compose -f $COMPOSE_FILE logs -f"
    echo "   ⏹️  Stop services:   docker-compose -f $COMPOSE_FILE down"
    echo "   🔄 Restart:         docker-compose -f $COMPOSE_FILE restart"
    echo "   📊 Service status:  docker-compose -f $COMPOSE_FILE ps"
    echo ""
    echo "🧪 Quick Test:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "📁 Log Locations:"
    echo "   📜 App logs:        ./logs/"
    echo "   🐳 Container logs:  docker-compose -f $COMPOSE_FILE logs"
    echo ""
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    docker-compose -f $COMPOSE_FILE down --remove-orphans
    log_info "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    docker-compose -f $COMPOSE_FILE down --volumes --remove-orphans
    docker system prune -f
    log_info "Cleanup completed"
}

# Show help
show_help() {
    echo "🚀 RAPTOR Production Deployment Script"
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
    echo "  --dry-run           Show what would be done without executing"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT         Deployment environment"
    echo "  CLEAN_BUILD         Clean build flag (true/false)"
    echo "  SKIP_BUILD          Skip build flag (true/false)"
    echo "  SKIP_HEALTH         Skip health checks (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Standard production deployment"
    echo "  $0 --env development                  # Development deployment"
    echo "  $0 --clean                           # Clean build deployment"
    echo "  $0 --rollback                        # Rollback current deployment"
    echo "  $0 --cleanup                         # Full cleanup"
    echo ""
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
    echo "🚀 RAPTOR Production Deployment"
    echo "Environment: $ENVIRONMENT"
    echo "================================"
    
    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "🧪 DRY RUN MODE - No changes will be made"
        echo "Would execute:"
        echo "1. Check prerequisites"
        echo "2. Create configuration files"
        echo "3. Build Docker images"
        echo "4. Deploy services"
        echo "5. Run health checks"
        exit 0
    fi
    
    # Execute deployment steps
    check_prerequisites
    create_configs
    create_docker_compose
    create_dockerfile
    
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

echo "✅ RAPTOR Production deployment completed successfully!"