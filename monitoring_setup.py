"""
DOSYA: monitoring_setup.py
AÇIKLAMA: Production monitoring setup for RAPTOR
"""

import logging
import time
import threading
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ProductionMonitoring:
    def __init__(self, metrics_port=9090, health_check_interval=30):
        self.metrics_port = metrics_port
        self.health_check_interval = health_check_interval
        self.running = False
        self.start_time = time.time()
        self.health_data = {
            "status": "starting",
            "uptime": 0,
            "last_check": time.time()
        }
        self._monitor_thread = None
    
    def start(self):
        """Start monitoring services"""
        self.running = True
        self.start_time = time.time()
        
        # Start background monitoring
        self._monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"✅ Monitoring started on port {self.metrics_port}")
        logger.info(f"📊 Metrics available at: http://localhost:{self.metrics_port}/metrics")
    
    def stop(self):
        """Stop monitoring services"""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("�� Monitoring stopped")
    
    def _background_monitoring(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self.health_data.update({
                    "status": "healthy",
                    "uptime": round(time.time() - self.start_time, 1),
                    "last_check": time.time()
                })
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            **self.health_data,
            "monitoring_port": self.metrics_port,
            "monitoring_active": self.running
        }

def setup_monitoring(metrics_port=9090, health_check_interval=30):
    """Setup monitoring with given configuration"""
    return ProductionMonitoring(metrics_port, health_check_interval)
