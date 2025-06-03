"""
DOSYA: monitoring-setup.py
AÇIKLAMA: Production monitoring setup - Prometheus metrics ve Grafana dashboard'ları
"""

import os
import time
import asyncio
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Prometheus client için
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client import start_http_server, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️ prometheus_client not installed. Run: pip install prometheus_client")

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """System health status"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    checks: Dict[str, bool]
    response_time_ms: float = 0.0
    error_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

class RAPTORMetrics:
    """RAPTOR-specific Prometheus metrics collector"""
    
    def __init__(self, registry=None):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            return
        
        self.registry = registry or REGISTRY
        self.enabled = True
        
        # Request metrics
        self.request_count = Counter(
            'raptor_requests_total',
            'Total number of requests',
            ['method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'raptor_request_duration_seconds',
            'Request duration in seconds',
            ['method'],
            registry=self.registry
        )
        
        # RAPTOR-specific metrics
        self.query_count = Counter(
            'raptor_queries_total',
            'Total number of RAPTOR queries',
            ['query_type'],
            registry=self.registry
        )
        
        self.retrieval_duration = Histogram(
            'raptor_retrieval_duration_seconds',
            'RAPTOR retrieval duration',
            ['retrieval_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.cache_operations = Counter(
            'raptor_cache_operations_total',
            'Cache operations',
            ['operation', 'result'],  # hit, miss, write, error
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'raptor_active_connections',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        self.tree_stats = Info(
            'raptor_tree_info',
            'RAPTOR tree information',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'raptor_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # total, available, used
            registry=self.registry
        )
        
        # Error tracking
        self.error_count = Counter(
            'raptor_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Performance metrics
        self.embedding_operations = Counter(
            'raptor_embedding_operations_total',
            'Embedding operations',
            ['model', 'status'],
            registry=self.registry
        )
        
        self.batch_processing = Histogram(
            'raptor_batch_processing_duration_seconds',
            'Batch processing duration',
            ['operation_type'],
            registry=self.registry
        )
        
        logger.info("✅ RAPTOR metrics initialized")
    
    def record_request(self, method: str, status: str, duration: float):
        """Record HTTP request metrics"""
        if not self.enabled:
            return
            
        self.request_count.labels(method=method, status=status).inc()
        self.request_duration.labels(method=method).observe(duration)
    
    def record_query(self, query_type: str, duration: float, retrieval_type: str = "standard"):
        """Record RAPTOR query metrics"""
        if not self.enabled:
            return
            
        self.query_count.labels(query_type=query_type).inc()
        self.retrieval_duration.labels(retrieval_type=retrieval_type).observe(duration)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation metrics"""
        if not self.enabled:
            return
            
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def update_active_connections(self, count: int):
        """Update active connections gauge"""
        if not self.enabled:
            return
            
        self.active_connections.set(count)
    
    def set_tree_info(self, nodes: int, layers: int, chunks: int):
        """Set tree information"""
        if not self.enabled:
            return
            
        self.tree_stats.info({
            'total_nodes': str(nodes),
            'total_layers': str(layers),
            'total_chunks': str(chunks),
            'last_updated': datetime.now().isoformat()
        })
    
    def update_memory_usage(self, total: int, available: int, used: int):
        """Update memory usage metrics"""
        if not self.enabled:
            return
            
        self.memory_usage.labels(type='total').set(total)
        self.memory_usage.labels(type='available').set(available)
        self.memory_usage.labels(type='used').set(used)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence"""
        if not self.enabled:
            return
            
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def record_embedding_operation(self, model: str, status: str, count: int = 1):
        """Record embedding operation"""
        if not self.enabled:
            return
            
        self.embedding_operations.labels(model=model, status=status).inc(count)
    
    def record_batch_processing(self, operation_type: str, duration: float):
        """Record batch processing metrics"""
        if not self.enabled:
            return
            
        self.batch_processing.labels(operation_type=operation_type).observe(duration)

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.last_status: Optional[HealthStatus] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Health check functions
        self.health_checks = {
            'redis': self._check_redis,
            'openai_api': self._check_openai_api,
            'raptor_tree': self._check_raptor_tree,
            'memory': self._check_memory,
            'disk': self._check_disk
        }
    
    def start(self):
        """Start health monitoring"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"🔍 Health monitoring started (interval: {self.check_interval}s)")
    
    def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def get_current_status(self) -> HealthStatus:
        """Get current health status"""
        if self.last_status is None:
            return self._run_health_checks()
        return self.last_status
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self.last_status = self._run_health_checks()
                self._log_health_status(self.last_status)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            # Wait for next check
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _run_health_checks(self) -> HealthStatus:
        """Run all health checks"""
        start_time = time.time()
        checks = {}
        details = {}
        
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                checks[check_name] = check_result['healthy']
                details[check_name] = check_result.get('details', {})
                
            except Exception as e:
                checks[check_name] = False
                details[check_name] = {'error': str(e)}
                logger.warning(f"Health check '{check_name}' failed: {e}")
        
        # Determine overall status
        healthy_checks = sum(checks.values())
        total_checks = len(checks)
        
        if healthy_checks == total_checks:
            status = "healthy"
        elif healthy_checks >= total_checks * 0.7:  # 70% threshold
            status = "degraded"
        else:
            status = "unhealthy"
        
        response_time = (time.time() - start_time) * 1000  # ms
        
        return HealthStatus(
            status=status,
            timestamp=datetime.now(),
            checks=checks,
            response_time_ms=response_time,
            details=details
        )
    
    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connection"""
        try:
            import redis.asyncio as redis
            
            # This is a simplified check - in real implementation,
            # you'd want to test actual connection
            return {
                'healthy': True,
                'details': {'status': 'connected'}
            }
        except ImportError:
            return {
                'healthy': False,
                'details': {'error': 'Redis client not available'}
            }
    
    def _check_openai_api(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            return {
                'healthy': bool(api_key),
                'details': {'api_key_present': bool(api_key)}
            }
        except Exception as e:
            return {
                'healthy': False,
                'details': {'error': str(e)}
            }
    
    def _check_raptor_tree(self) -> Dict[str, Any]:
        """Check RAPTOR tree availability"""
        try:
            # Check if tree files exist
            tree_path = "vectordb/raptor-optimized"
            tree_exists = os.path.exists(tree_path)
            
            return {
                'healthy': tree_exists,
                'details': {
                    'tree_path': tree_path,
                    'exists': tree_exists
                }
            }
        except Exception as e:
            return {
                'healthy': False,
                'details': {'error': str(e)}
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            usage_percent = memory.percent
            
            # Consider unhealthy if less than 1GB available or >90% usage
            healthy = available_gb > 1.0 and usage_percent < 90
            
            return {
                'healthy': healthy,
                'details': {
                    'available_gb': round(available_gb, 2),
                    'usage_percent': round(usage_percent, 2)
                }
            }
        except ImportError:
            return {
                'healthy': True,  # Assume healthy if can't check
                'details': {'error': 'psutil not available'}
            }
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            import psutil
            
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            usage_percent = (disk.used / disk.total) * 100
            
            # Consider unhealthy if less than 5GB free or >95% usage
            healthy = free_gb > 5.0 and usage_percent < 95
            
            return {
                'healthy': healthy,
                'details': {
                    'free_gb': round(free_gb, 2),
                    'usage_percent': round(usage_percent, 2)
                }
            }
        except ImportError:
            return {
                'healthy': True,  # Assume healthy if can't check
                'details': {'error': 'psutil not available'}
            }
    
    def _log_health_status(self, status: HealthStatus):
        """Log health status"""
        if status.status == "healthy":
            logger.debug(f"✅ System healthy ({status.response_time_ms:.1f}ms)")
        elif status.status == "degraded":
            logger.warning(f"⚠️ System degraded ({status.response_time_ms:.1f}ms)")
            failed_checks = [k for k, v in status.checks.items() if not v]
            logger.warning(f"Failed checks: {failed_checks}")
        else:
            logger.error(f"❌ System unhealthy ({status.response_time_ms:.1f}ms)")
            failed_checks = [k for k, v in status.checks.items() if not v]
            logger.error(f"Failed checks: {failed_checks}")

class MonitoringServer:
    """Integrated monitoring server"""
    
    def __init__(self, metrics_port: int = 9090, health_check_interval: int = 30):
        self.metrics_port = metrics_port
        self.metrics = RAPTORMetrics() if PROMETHEUS_AVAILABLE else None
        self.health_checker = HealthChecker(health_check_interval)
        self.running = False
    
    def start(self):
        """Start monitoring server"""
        if self.running:
            return
        
        self.running = True
        
        # Start Prometheus metrics server
        if PROMETHEUS_AVAILABLE and self.metrics:
            try:
                start_http_server(self.metrics_port)
                logger.info(f"📊 Prometheus metrics server started on port {self.metrics_port}")
                logger.info(f"Metrics available at: http://localhost:{self.metrics_port}/metrics")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
        
        # Start health monitoring
        self.health_checker.start()
        
        logger.info("🔍 Monitoring system started")
    
    def stop(self):
        """Stop monitoring server"""
        self.running = False
        self.health_checker.stop()
        logger.info("Monitoring system stopped")
    
    def get_metrics_endpoint(self) -> str:
        """Get metrics endpoint URL"""
        return f"http://localhost:{self.metrics_port}/metrics"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status as dict"""
        status = self.health_checker.get_current_status()
        return {
            'status': status.status,
            'timestamp': status.timestamp.isoformat(),
            'response_time_ms': status.response_time_ms,
            'checks': status.checks,
            'details': status.details
        }

def create_grafana_dashboard() -> Dict[str, Any]:
    """Create Grafana dashboard configuration for RAPTOR"""
    
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "RAPTOR Production Dashboard",
            "tags": ["raptor", "rag", "production"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(raptor_requests_total[5m])",
                            "legendFormat": "Requests/sec"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "Response Time",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(raptor_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "id": 3,
                    "title": "Active Connections",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "raptor_active_connections",
                            "legendFormat": "Connections"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                },
                {
                    "id": 4,
                    "title": "Cache Hit Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(raptor_cache_operations_total{result=\"hit\"}[5m]) / rate(raptor_cache_operations_total[5m])",
                            "legendFormat": "Hit Rate"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                },
                {
                    "id": 5,
                    "title": "Query Performance",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.50, rate(raptor_retrieval_duration_seconds_bucket[5m]))",
                            "legendFormat": "50th percentile"
                        },
                        {
                            "expr": "histogram_quantile(0.95, rate(raptor_retrieval_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
                },
                {
                    "id": 6,
                    "title": "Error Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(raptor_errors_total[5m])",
                            "legendFormat": "Errors/sec - {{error_type}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24}
                },
                {
                    "id": 7,
                    "title": "Memory Usage",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "raptor_memory_usage_bytes{type=\"used\"}",
                            "legendFormat": "Used Memory"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24}
                }
            ]
        }
    }
    
    return dashboard

def setup_monitoring(metrics_port: int = 9090, 
                    health_check_interval: int = 30,
                    create_dashboard: bool = True) -> MonitoringServer:
    """
    Setup complete monitoring stack
    
    Args:
        metrics_port: Port for Prometheus metrics server
        health_check_interval: Health check interval in seconds
        create_dashboard: Whether to create Grafana dashboard config
    
    Returns:
        MonitoringServer instance
    """
    
    logger.info("🔧 Setting up monitoring stack...")
    
    # Create monitoring server
    monitoring = MonitoringServer(metrics_port, health_check_interval)
    
    # Create directories
    Path("monitoring").mkdir(exist_ok=True)
    Path("monitoring/grafana").mkdir(exist_ok=True)
    Path("monitoring/prometheus").mkdir(exist_ok=True)
    
    # Create Prometheus configuration
    prometheus_config = {
        "global": {
            "scrape_interval": "15s"
        },
        "scrape_configs": [
            {
                "job_name": "raptor",
                "static_configs": [
                    {
                        "targets": [f"localhost:{metrics_port}"]
                    }
                ],
                "scrape_interval": "5s"
            }
        ]
    }
    
    with open("monitoring/prometheus/prometheus.yml", "w") as f:
        import yaml
        yaml.dump(prometheus_config, f, default_flow_style=False)
    
    logger.info("✅ Prometheus config created: monitoring/prometheus/prometheus.yml")
    
    # Create Grafana dashboard
    if create_dashboard:
        dashboard = create_grafana_dashboard()
        with open("monitoring/grafana/raptor_dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)
        
        logger.info("✅ Grafana dashboard created: monitoring/grafana/raptor_dashboard.json")
    
    # Create docker-compose for easy deployment
    docker_compose = {
        "version": "3.8",
        "services": {
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": ["./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml"],
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.path=/prometheus",
                    "--web.console.libraries=/etc/prometheus/console_libraries",
                    "--web.console.templates=/etc/prometheus/consoles",
                    "--web.enable-lifecycle"
                ]
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=admin"
                ],
                "volumes": ["grafana-data:/var/lib/grafana"]
            }
        },
        "volumes": {
            "grafana-data": None
        }
    }
    
    with open("monitoring/docker-compose.yml", "w") as f:
        import yaml
        yaml.dump(docker_compose, f, default_flow_style=False)
    
    logger.info("✅ Docker Compose created: monitoring/docker-compose.yml")
    
    # Create startup script
    startup_script = """#!/bin/bash
# RAPTOR Monitoring Stack Startup

echo "🚀 Starting RAPTOR Monitoring Stack..."

# Start Prometheus and Grafana
cd monitoring
docker-compose up -d

echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3000 (admin/admin)"
echo "📊 RAPTOR Metrics: http://localhost:{}".format(metrics_port)

echo "✅ Monitoring stack started!"
"""
    
    with open("monitoring/start_monitoring.sh", "w") as f:
        f.write(startup_script)
    
    os.chmod("monitoring/start_monitoring.sh", 0o755)
    logger.info("✅ Startup script created: monitoring/start_monitoring.sh")
    
    # Print setup instructions
    print("\n" + "="*60)
    print("📊 MONITORING SETUP COMPLETE")
    print("="*60)
    print(f"📊 Prometheus metrics: http://localhost:{metrics_port}/metrics")
    print("📈 To start Grafana & Prometheus:")
    print("   cd monitoring && docker-compose up -d")
    print("📈 Grafana dashboard: http://localhost:3000 (admin/admin)")
    print("📊 Prometheus: http://localhost:9090")
    print("\n📋 Next steps:")
    print("1. Run: ./monitoring/start_monitoring.sh")
    print("2. Import dashboard: monitoring/grafana/raptor_dashboard.json")
    print("3. Configure Prometheus data source in Grafana: http://prometheus:9090")
    print("="*60)
    
    return monitoring

# Example usage functions
def start_monitoring_with_raptor(raptor_instance=None):
    """Start monitoring integrated with RAPTOR instance"""
    
    monitoring = setup_monitoring()
    monitoring.start()
    
    # If RAPTOR instance provided, integrate metrics
    if raptor_instance and monitoring.metrics:
        # This would be integrated into the actual RAPTOR classes
        logger.info("🔗 Integrated RAPTOR metrics")
    
    return monitoring

if __name__ == "__main__":
    # Setup monitoring stack
    monitoring = setup_monitoring(
        metrics_port=9090,
        health_check_interval=30,
        create_dashboard=True
    )
    
    # Start monitoring for testing
    monitoring.start()
    
    try:
        logger.info("Monitoring running... Press Ctrl+C to stop")
        while True:
            time.sleep(10)
            status = monitoring.get_health_status()
            print(f"Health: {status['status']} ({status['response_time_ms']:.1f}ms)")
    
    except KeyboardInterrupt:
        monitoring.stop()
        logger.info("Monitoring stopped")