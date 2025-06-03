"""
DOSYA: deploy-raptor-production.py
AÇIKLAMA: Production deployment script - Comprehensive deployment automation
"""

import os
import sys
import time
import shutil
import subprocess
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import signal
from contextlib import contextmanager

# Import our configuration
from production_config import get_production_config, Environment
from monitoring_setup import setup_monitoring

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentResult:
    """Deployment operation result"""
    success: bool
    message: str
    details: Dict[str, Any]
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'details': self.details,
            'duration_seconds': self.duration_seconds
        }

class DeploymentValidator:
    """Pre-deployment validation"""
    
    @staticmethod
    def validate_system_requirements() -> DeploymentResult:
        """Validate system meets production requirements"""
        start_time = time.time()
        checks = {}
        details = {}
        
        try:
            # Memory check (minimum 8GB for production)
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            checks['memory'] = memory_gb >= 8.0
            details['memory'] = {
                'total_gb': round(memory_gb, 2),
                'required_gb': 8.0,
                'available_gb': round(memory.available / (1024**3), 2)
            }
            
            # CPU check (minimum 4 cores recommended)
            cpu_count = psutil.cpu_count()
            checks['cpu'] = cpu_count >= 4
            details['cpu'] = {
                'cores': cpu_count,
                'recommended': 4
            }
            
            # Disk space check (minimum 50GB free)
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            checks['disk'] = free_gb >= 50.0
            details['disk'] = {
                'free_gb': round(free_gb, 2),
                'required_gb': 50.0
            }
            
            # Python version check
            python_version = sys.version_info
            checks['python'] = python_version >= (3, 8)
            details['python'] = {
                'current': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'required': "3.8+"
            }
            
            # Required packages check
            required_packages = [
                'torch', 'transformers', 'openai', 'redis', 
                'fastapi', 'uvicorn', 'prometheus_client'
            ]
            
            package_status = {}
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                    package_status[package] = True
                except ImportError:
                    package_status[package] = False
            
            checks['packages'] = all(package_status.values())
            details['packages'] = package_status
            
            # Environment variables check
            required_env_vars = ['OPENAI_API_KEY']
            env_status = {var: bool(os.getenv(var)) for var in required_env_vars}
            checks['environment'] = all(env_status.values())
            details['environment'] = env_status
            
            # Overall validation
            all_passed = all(checks.values())
            
            return DeploymentResult(
                success=all_passed,
                message="System validation passed" if all_passed else "System validation failed",
                details={'checks': checks, 'details': details},
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Validation error: {str(e)}",
                details={'error': str(e)},
                duration_seconds=time.time() - start_time
            )
    
    @staticmethod
    def validate_raptor_tree(tree_path: str) -> DeploymentResult:
        """Validate RAPTOR tree availability and integrity"""
        start_time = time.time()
        
        try:
            if not os.path.exists(tree_path):
                return DeploymentResult(
                    success=False,
                    message=f"RAPTOR tree not found at {tree_path}",
                    details={'tree_path': tree_path, 'exists': False},
                    duration_seconds=time.time() - start_time
                )
            
            # Check file size (should be substantial)
            file_size = os.path.getsize(tree_path)
            size_mb = file_size / (1024 * 1024)
            
            if size_mb < 1:  # Less than 1MB might indicate corrupt tree
                return DeploymentResult(
                    success=False,
                    message=f"RAPTOR tree file too small ({size_mb:.2f}MB)",
                    details={'tree_path': tree_path, 'size_mb': size_mb},
                    duration_seconds=time.time() - start_time
                )
            
            # Try to load tree (basic check)
            try:
                import pickle
                with open(tree_path, 'rb') as f:
                    tree_data = pickle.load(f)
                
                # Basic structure validation
                tree_valid = hasattr(tree_data, 'all_nodes') and hasattr(tree_data, 'num_layers')
                
                return DeploymentResult(
                    success=tree_valid,
                    message="RAPTOR tree validation passed" if tree_valid else "RAPTOR tree structure invalid",
                    details={
                        'tree_path': tree_path,
                        'size_mb': round(size_mb, 2),
                        'structure_valid': tree_valid,
                        'num_layers': getattr(tree_data, 'num_layers', 'unknown'),
                        'node_count': len(getattr(tree_data, 'all_nodes', {}))
                    },
                    duration_seconds=time.time() - start_time
                )
                
            except Exception as e:
                return DeploymentResult(
                    success=False,
                    message=f"Failed to load RAPTOR tree: {str(e)}",
                    details={'tree_path': tree_path, 'load_error': str(e)},
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Tree validation error: {str(e)}",
                details={'error': str(e)},
                duration_seconds=time.time() - start_time
            )

class ServiceManager:
    """Production service management"""
    
    def __init__(self, config_env: str = "production"):
        self.config = get_production_config(config_env)
        self.process: Optional[subprocess.Popen] = None
        self.monitoring = None
    
    def start_redis(self) -> DeploymentResult:
        """Start Redis server"""
        start_time = time.time()
        
        try:
            # Check if Redis is already running
            try:
                import redis
                r = redis.Redis(
                    host=self.config.database.redis_host,
                    port=self.config.database.redis_port,
                    password=self.config.database.redis_password,
                    socket_connect_timeout=5
                )
                r.ping()
                
                return DeploymentResult(
                    success=True,
                    message="Redis already running",
                    details={'status': 'already_running'},
                    duration_seconds=time.time() - start_time
                )
                
            except (redis.ConnectionError, redis.TimeoutError):
                # Redis not running, try to start it
                pass
            
            # Try to start Redis
            redis_config = f"""
# RAPTOR Production Redis Configuration
port {self.config.database.redis_port}
bind 127.0.0.1
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
"""
            
            # Write Redis config
            redis_config_path = "redis-raptor.conf"
            with open(redis_config_path, 'w') as f:
                f.write(redis_config)
            
            # Start Redis server
            redis_cmd = f"redis-server {redis_config_path}"
            subprocess.Popen(redis_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for Redis to start
            time.sleep(3)
            
            # Verify Redis is running
            try:
                r = redis.Redis(
                    host=self.config.database.redis_host,
                    port=self.config.database.redis_port,
                    socket_connect_timeout=5
                )
                r.ping()
                
                return DeploymentResult(
                    success=True,
                    message="Redis started successfully",
                    details={'status': 'started', 'config_file': redis_config_path},
                    duration_seconds=time.time() - start_time
                )
                
            except Exception as e:
                return DeploymentResult(
                    success=False,
                    message=f"Failed to verify Redis startup: {str(e)}",
                    details={'error': str(e)},
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Redis startup failed: {str(e)}",
                details={'error': str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def start_monitoring(self) -> DeploymentResult:
        """Start monitoring stack"""
        start_time = time.time()
        
        try:
            self.monitoring = setup_monitoring(
                metrics_port=self.config.monitoring.prometheus_port,
                health_check_interval=self.config.monitoring.health_check_interval
            )
            
            self.monitoring.start()
            
            return DeploymentResult(
                success=True,
                message="Monitoring started successfully",
                details={
                    'prometheus_port': self.config.monitoring.prometheus_port,
                    'metrics_url': f"http://localhost:{self.config.monitoring.prometheus_port}/metrics"
                },
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Monitoring startup failed: {str(e)}",
                details={'error': str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def start_raptor_server(self, script_path: str = "generic-qa-server.py") -> DeploymentResult:
        """Start RAPTOR server"""
        start_time = time.time()
        
        try:
            if not os.path.exists(script_path):
                return DeploymentResult(
                    success=False,
                    message=f"RAPTOR server script not found: {script_path}",
                    details={'script_path': script_path},
                    duration_seconds=time.time() - start_time
                )
            
            # Prepare environment variables
            env = os.environ.copy()
            env.update({
                'RAPTOR_ENV': self.config.environment.value,
                'SERVER_HOST': self.config.server_host,
                'SERVER_PORT': str(self.config.server_port),
                'REDIS_HOST': self.config.database.redis_host,
                'REDIS_PORT': str(self.config.database.redis_port),
                'REDIS_PASSWORD': self.config.database.redis_password or '',
            })
            
            # Start server using uvicorn for production
            if self.config.environment == Environment.PRODUCTION:
                cmd = [
                    sys.executable, "-m", "uvicorn",
                    f"{Path(script_path).stem}:app",
                    "--host", self.config.server_host,
                    "--port", str(self.config.server_port),
                    "--workers", str(self.config.workers),
                    "--timeout-keep-alive", str(self.config.worker_timeout),
                    "--log-level", "info"
                ]
            else:
                # Development/staging - simpler startup
                cmd = [sys.executable, script_path]
            
            logger.info(f"Starting RAPTOR server: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for startup
            time.sleep(5)
            
            # Check if process is still running
            if self.process.poll() is None:
                return DeploymentResult(
                    success=True,
                    message=f"RAPTOR server started (PID: {self.process.pid})",
                    details={
                        'pid': self.process.pid,
                        'host': self.config.server_host,
                        'port': self.config.server_port,
                        'workers': self.config.workers,
                        'url': f"http://{self.config.server_host}:{self.config.server_port}"
                    },
                    duration_seconds=time.time() - start_time
                )
            else:
                # Process died, get error output
                stdout, stderr = self.process.communicate()
                return DeploymentResult(
                    success=False,
                    message="RAPTOR server failed to start",
                    details={
                        'stdout': stdout.decode() if stdout else '',
                        'stderr': stderr.decode() if stderr else '',
                        'return_code': self.process.returncode
                    },
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Server startup failed: {str(e)}",
                details={'error': str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def stop_services(self) -> DeploymentResult:
        """Stop all services"""
        start_time = time.time()
        stopped_services = []
        
        try:
            # Stop RAPTOR server
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                    stopped_services.append("raptor_server")
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                    stopped_services.append("raptor_server (force)")
            
            # Stop monitoring
            if self.monitoring:
                self.monitoring.stop()
                stopped_services.append("monitoring")
            
            return DeploymentResult(
                success=True,
                message=f"Services stopped: {', '.join(stopped_services)}",
                details={'stopped_services': stopped_services},
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Error stopping services: {str(e)}",
                details={'error': str(e), 'stopped_services': stopped_services},
                duration_seconds=time.time() - start_time
            )

class ProductionDeployer:
    """Main production deployment orchestrator"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = get_production_config(environment)
        self.service_manager = ServiceManager(environment)
        self.deployment_log = []
    
    def log_step(self, step: str, result: DeploymentResult):
        """Log deployment step"""
        self.deployment_log.append({
            'step': step,
            'timestamp': time.time(),
            'result': result.to_dict()
        })
        
        if result.success:
            logger.info(f"✅ {step}: {result.message}")
        else:
            logger.error(f"❌ {step}: {result.message}")
    
    def deploy(self, tree_path: str = "vectordb/raptor-optimized", 
               skip_validation: bool = False) -> DeploymentResult:
        """Full production deployment"""
        deployment_start = time.time()
        
        logger.info("🚀 Starting RAPTOR production deployment...")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        try:
            # Step 1: System validation
            if not skip_validation:
                logger.info("🔍 Step 1: System validation...")
                validation_result = DeploymentValidator.validate_system_requirements()
                self.log_step("System Validation", validation_result)
                
                if not validation_result.success:
                    return DeploymentResult(
                        success=False,
                        message="System validation failed",
                        details={'validation': validation_result.details},
                        duration_seconds=time.time() - deployment_start
                    )
            
            # Step 2: RAPTOR tree validation
            logger.info("🌳 Step 2: RAPTOR tree validation...")
            tree_result = DeploymentValidator.validate_raptor_tree(tree_path)
            self.log_step("RAPTOR Tree Validation", tree_result)
            
            if not tree_result.success:
                return DeploymentResult(
                    success=False,
                    message="RAPTOR tree validation failed",
                    details={'tree_validation': tree_result.details},
                    duration_seconds=time.time() - deployment_start
                )
            
            # Step 3: Start Redis
            logger.info("🔧 Step 3: Starting Redis...")
            redis_result = self.service_manager.start_redis()
            self.log_step("Redis Startup", redis_result)
            
            if not redis_result.success:
                return DeploymentResult(
                    success=False,
                    message="Redis startup failed",
                    details={'redis': redis_result.details},
                    duration_seconds=time.time() - deployment_start
                )
            
            # Step 4: Start monitoring
            logger.info("📊 Step 4: Starting monitoring...")
            monitoring_result = self.service_manager.start_monitoring()
            self.log_step("Monitoring Startup", monitoring_result)
            
            if not monitoring_result.success:
                logger.warning("⚠️ Monitoring startup failed, continuing deployment...")
            
            # Step 5: Start RAPTOR server
            logger.info("🚀 Step 5: Starting RAPTOR server...")
            server_result = self.service_manager.start_raptor_server()
            self.log_step("RAPTOR Server Startup", server_result)
            
            if not server_result.success:
                return DeploymentResult(
                    success=False,
                    message="RAPTOR server startup failed",
                    details={'server': server_result.details},
                    duration_seconds=time.time() - deployment_start
                )
            
            # Step 6: Health check
            logger.info("🔍 Step 6: Final health check...")
            time.sleep(10)  # Wait for services to stabilize
            
            health_status = self.service_manager.monitoring.get_health_status() if self.service_manager.monitoring else {'status': 'unknown'}
            
            # Save deployment log
            self._save_deployment_log()
            
            # Success!
            total_time = time.time() - deployment_start
            
            success_message = f"🎉 RAPTOR production deployment successful! ({total_time:.1f}s)"
            logger.info(success_message)
            
            return DeploymentResult(
                success=True,
                message="Production deployment completed successfully",
                details={
                    'server_url': f"http://{self.config.server_host}:{self.config.server_port}",
                    'health_check_url': f"http://{self.config.server_host}:{self.config.server_port}/health",
                    'metrics_url': f"http://localhost:{self.config.monitoring.prometheus_port}/metrics",
                    'environment': self.environment,
                    'tree_path': tree_path,
                    'health_status': health_status,
                    'deployment_log': self.deployment_log
                },
                duration_seconds=total_time
            )
            
        except Exception as e:
            logger.error(f"💥 Deployment failed: {str(e)}")
            
            # Attempt cleanup
            try:
                self.service_manager.stop_services()
            except:
                pass
            
            return DeploymentResult(
                success=False,
                message=f"Deployment failed: {str(e)}",
                details={'error': str(e), 'deployment_log': self.deployment_log},
                duration_seconds=time.time() - deployment_start
            )
    
    def _save_deployment_log(self):
        """Save deployment log to file"""
        log_file = f"deployment_log_{self.environment}_{int(time.time())}.json"
        
        with open(log_file, 'w') as f:
            json.dump({
                'environment': self.environment,
                'timestamp': time.time(),
                'deployment_log': self.deployment_log
            }, f, indent=2)
        
        logger.info(f"📝 Deployment log saved: {log_file}")
    
    def rollback(self) -> DeploymentResult:
        """Rollback deployment"""
        logger.info("⏪ Rolling back deployment...")
        
        return self.service_manager.stop_services()

def signal_handler(signum, frame, deployer):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down...")
    deployer.rollback()
    sys.exit(0)

def main():
    """Main deployment entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAPTOR Production Deployer")
    parser.add_argument("--env", choices=["development", "staging", "production"], 
                       default="production", help="Deployment environment")
    parser.add_argument("--tree-path", default="vectordb/raptor-optimized",
                       help="Path to RAPTOR tree")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip system validation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate but don't deploy")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = ProductionDeployer(args.env)
    
    # Setup signal handlers
    import functools
    signal.signal(signal.SIGINT, functools.partial(signal_handler, deployer=deployer))
    signal.signal(signal.SIGTERM, functools.partial(signal_handler, deployer=deployer))
    
    if args.dry_run:
        # Just run validation
        logger.info("🧪 Dry run - validation only")
        
        system_result = DeploymentValidator.validate_system_requirements()
        tree_result = DeploymentValidator.validate_raptor_tree(args.tree_path)
        
        print("\n" + "="*60)
        print("🧪 DRY RUN RESULTS")
        print("="*60)
        print(f"System Validation: {'✅ PASS' if system_result.success else '❌ FAIL'}")
        print(f"Tree Validation: {'✅ PASS' if tree_result.success else '❌ FAIL'}")
        
        if not system_result.success:
            print(f"System Issues: {system_result.message}")
        if not tree_result.success:
            print(f"Tree Issues: {tree_result.message}")
        
        print("="*60)
        sys.exit(0 if system_result.success and tree_result.success else 1)
    
    # Run deployment
    result = deployer.deploy(args.tree_path, args.skip_validation)
    
    if result.success:
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT SUCCESSFUL")
        print("="*60)
        print(f"🌐 Server: {result.details.get('server_url')}")
        print(f"🔍 Health: {result.details.get('health_check_url')}")
        print(f"📊 Metrics: {result.details.get('metrics_url')}")
        print("="*60)
        
        # Keep running
        try:
            logger.info("🔄 Deployment running... Press Ctrl+C to stop")
            while True:
                time.sleep(30)
                if deployer.service_manager.monitoring:
                    status = deployer.service_manager.monitoring.get_health_status()
                    logger.info(f"Health: {status['status']}")
        except KeyboardInterrupt:
            deployer.rollback()
    else:
        print("\n" + "="*60)
        print("❌ DEPLOYMENT FAILED")
        print("="*60)
        print(f"Error: {result.message}")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()