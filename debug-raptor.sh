#!/bin/bash

echo "🐳 Docker RAPTOR Debug Script"
echo "============================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 1. Deploy without health check
log_step "Step 1: Deploying services without health check..."
./scripts/deploy.sh --no-health

if [ $? -ne 0 ]; then
    log_error "Deployment failed"
    exit 1
fi

# Wait for containers to start
log_info "Waiting for containers to start (30 seconds)..."
sleep 30

# 2. Check container status
log_step "Step 2: Checking container status..."
echo "Container status:"
docker-compose -f docker-compose.production.yml ps

# 3. Check if container is running
if ! docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "raptor-app"; then
    log_error "RAPTOR app container is not running!"
    log_info "Checking why container exited..."
    docker logs raptor-app
    exit 1
fi

log_info "✅ RAPTOR app container is running"

# 4. Check container logs
log_step "Step 3: Checking container logs..."
echo "=== CONTAINER LOGS (last 50 lines) ==="
docker logs --tail 50 raptor-app
echo "=== END LOGS ==="

# 5. Test container internals
log_step "Step 4: Testing container internals..."

# Check if files exist in container
log_info "Checking files in container..."
docker exec raptor-app ls -la /app/production_config.py 2>/dev/null
if [ $? -eq 0 ]; then
    log_info "✅ production_config.py exists in container"
else
    log_error "❌ production_config.py missing in container"
fi

docker exec raptor-app ls -la /app/vectordb/ 2>/dev/null
if [ $? -eq 0 ]; then
    log_info "✅ vectordb directory exists in container"
else
    log_error "❌ vectordb directory missing in container"
fi

# Check environment variables
log_info "Checking environment variables in container..."
echo "Environment variables:"
docker exec raptor-app env | grep -E "(OPENAI|REDIS|RAPTOR|SERVER)" | head -10

# 6. Test health endpoint manually
log_step "Step 5: Testing health endpoint..."

# Test from host
log_info "Testing health endpoint from host..."
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    log_info "✅ Health endpoint accessible from host"
    response=$(curl -s http://localhost:8000/health)
    echo "Response: $response"
else
    log_warn "❌ Health endpoint NOT accessible from host"
    
    # Test from inside container
    log_info "Testing health endpoint from inside container..."
    docker exec raptor-app curl -f -s http://localhost:8000/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_info "✅ Health endpoint accessible from inside container"
        response=$(docker exec raptor-app curl -s http://localhost:8000/health)
        echo "Internal response: $response"
    else
        log_error "❌ Health endpoint NOT accessible from inside container"
    fi
fi

# 7. Check if server is actually running inside container
log_step "Step 6: Checking if server process is running..."
docker exec raptor-app ps aux | grep -E "(python|uvicorn)" | grep -v grep
if [ $? -eq 0 ]; then
    log_info "✅ Python/server process found in container"
else
    log_error "❌ No Python/server process found in container"
fi

# 8. Check port binding
log_step "Step 7: Checking port binding..."
if netstat -tln 2>/dev/null | grep -q ":8000 " || ss -tln 2>/dev/null | grep -q ":8000 "; then
    log_info "✅ Port 8000 is bound on host"
else
    log_warn "❌ Port 8000 is NOT bound on host"
fi

# 9. Try manual startup inside container
log_step "Step 8: Testing manual startup inside container..."
log_info "Attempting manual server start inside container..."

# Kill existing server if any
docker exec raptor-app pkill -f "generic-qa-server.py" 2>/dev/null || true
docker exec raptor-app pkill -f "uvicorn" 2>/dev/null || true
sleep 5

# Start server manually and test
log_info "Starting server manually in background..."
docker exec -d raptor-app python /app/generic-qa-server.py

# Wait and test
sleep 15
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    log_info "✅ Manual startup successful!"
    response=$(curl -s http://localhost:8000/health)
    echo "Health response: $response"
    
    # Show solution
    echo ""
    echo "🎯 SOLUTION FOUND!"
    echo "=================="
    log_info "Manual startup works - there might be a startup timing issue"
    log_info "Try increasing the health check start period in docker-compose.yml"
    
else
    log_error "❌ Manual startup also failed"
    echo ""
    echo "🔍 Additional Debug Information:"
    echo "================================"
    
    # Show more detailed logs
    echo "Full container logs:"
    docker logs raptor-app
    
    echo ""
    echo "Container processes:"
    docker exec raptor-app ps aux
    
    echo ""
    echo "Container network:"
    docker exec raptor-app netstat -tln 2>/dev/null || docker exec raptor-app ss -tln
fi

# 10. Cleanup or keep running
echo ""
echo "🔧 Options:"
echo "==========="
echo "1. Keep containers running for manual debug:"
echo "   docker exec -it raptor-app bash"
echo ""
echo "2. View live logs:"
echo "   docker logs -f raptor-app"
echo ""
echo "3. Stop containers:"
echo "   docker-compose -f docker-compose.production.yml down"
echo ""

read -p "Do you want to stop containers now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Stopping containers..."
    docker-compose -f docker-compose.production.yml down
    log_info "Containers stopped"
else
    log_info "Containers left running for manual debug"
fi

echo "✅ Debug completed!"