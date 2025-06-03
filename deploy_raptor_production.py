"""
DOSYA: enhanced-deploy-raptor-production.py
AÇIKLAMA: Enhanced production deployment script with automatic Redis setup
"""

import os
import sys
import time
import shutil
import subprocess
import psutil
import logging
import platform
import urllib.request
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import signal
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

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

class RedisManager:
    """Automatic Redis installation and management"""
    
    def __init__(self):
        self.redis_process = None
        self.redis_port = 6379
        self.redis_password = "Ph4nt0m4+4"
        self.redis_dir = Path("redis_local")
        self.config_file = "redis-raptor.conf"
    
    def check_redis_available(self) -> Tuple[bool, str, Dict]:
        """Check if Redis is available (system, docker, or custom)"""
        methods = []
        
        # Method 1: System Redis
        if shutil.which("redis-server"):
            try:
                import redis
                r = redis.Redis(host='localhost', port=self.redis_port, 
                              password=None, socket_connect_timeout=2)
                r.ping()
                methods.append(("system", "Redis already running on system", {"host": "localhost", "port": self.redis_port}))
            except:
                methods.append(("system_binary", "Redis binary available but not running", {"binary": shutil.which("redis-server")}))
        
        # Method 2: Docker Redis
        if shutil.which("docker"):
            try:
                result = subprocess.run(["docker", "ps", "--filter", "name=redis", "--format", "{{.Names}}"], 
                                      capture_output=True, text=True, timeout=10)
                if "redis" in result.stdout:
                    methods.append(("docker_running", "Redis container already running", {"container": "redis"}))
                else:
                    methods.append(("docker_available", "Docker available for Redis", {"docker": shutil.which("docker")}))
            except:
                pass
        
        if methods:
            return True, methods[0][0], {"methods": methods}
        else:
            return False, "no_redis", {"methods": []}
    
    def install_redis_system(self) -> DeploymentResult:
        """Install Redis using system package manager"""
        start_time = time.time()
        
        try:
            system = platform.system().lower()
            
            if system == "linux":
                # Ubuntu/Debian
                if shutil.which("apt"):
                    logger.info("📦 Installing Redis via apt...")
                    commands = [
                        ["sudo", "apt", "update"],
                        ["sudo", "apt", "install", "-y", "redis-server"]
                    ]
                # CentOS/RHEL
                elif shutil.which("yum"):
                    logger.info("📦 Installing Redis via yum...")
                    commands = [
                        ["sudo", "yum", "install", "-y", "epel-release"],
                        ["sudo", "yum", "install", "-y", "redis"]
                    ]
                # Fedora
                elif shutil.which("dnf"):
                    logger.info("📦 Installing Redis via dnf...")
                    commands = [
                        ["sudo", "dnf", "install", "-y", "redis"]
                    ]
                else:
                    return DeploymentResult(
                        success=False,
                        message="Unsupported Linux distribution for automatic Redis installation",
                        details={"system": system},
                        duration_seconds=time.time() - start_time
                    )
                
                # Execute installation commands
                for cmd in commands:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        return DeploymentResult(
                            success=False,
                            message=f"Failed to execute: {' '.join(cmd)}",
                            details={"error": result.stderr, "command": cmd},
                            duration_seconds=time.time() - start_time
                        )
                
                return DeploymentResult(
                    success=True,
                    message="Redis installed successfully via system package manager",
                    details={"method": "system_package", "system": system},
                    duration_seconds=time.time() - start_time
                )
            
            elif system == "darwin":  # macOS
                if shutil.which("brew"):
                    logger.info("📦 Installing Redis via Homebrew...")
                    result = subprocess.run(["brew", "install", "redis"], capture_output=True, text=True)
                    if result.returncode == 0:
                        return DeploymentResult(
                            success=True,
                            message="Redis installed successfully via Homebrew",
                            details={"method": "homebrew"},
                            duration_seconds=time.time() - start_time
                        )
                
                return DeploymentResult(
                    success=False,
                    message="Homebrew not available for Redis installation on macOS",
                    details={"system": system},
                    duration_seconds=time.time() - start_time
                )
            
            else:
                return DeploymentResult(
                    success=False,
                    message=f"Automatic Redis installation not supported on {system}",
                    details={"system": system},
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Redis system installation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def setup_redis_docker(self) -> DeploymentResult:
        """Setup Redis using Docker"""
        start_time = time.time()
        
        try:
            logger.info("🐳 Setting up Redis with Docker...")
            
            # Check if Redis container already exists
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=redis-raptor", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if "redis-raptor" in result.stdout:
                logger.info("♻️ Removing existing Redis container...")
                subprocess.run(["docker", "rm", "-f", "redis-raptor"], capture_output=True)
            
            # Start Redis container
            docker_cmd = [
                "docker", "run", "-d",
                "--name", "redis-raptor",
                "-p", f"{self.redis_port}:6379",
                "--restart", "unless-stopped",
                "redis:7-alpine",
                "redis-server",
                "--requirepass", self.redis_password,
                "--maxmemory", "1gb",
                "--maxmemory-policy", "allkeys-lru"
            ]
            
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return DeploymentResult(
                    success=False,
                    message="Failed to start Redis Docker container",
                    details={"error": result.stderr, "command": docker_cmd},
                    duration_seconds=time.time() - start_time
                )
            
            # Wait for Redis to start
            time.sleep(5)
            
            # Test connection
            test_result = subprocess.run(
                ["docker", "exec", "redis-raptor", "redis-cli", "-a", self.redis_password, "ping"],
                capture_output=True, text=True, timeout=10
            )
            
            if "PONG" not in test_result.stdout:
                return DeploymentResult(
                    success=False,
                    message="Redis Docker container started but not responding",
                    details={"test_output": test_result.stdout},
                    duration_seconds=time.time() - start_time
                )
            
            return DeploymentResult(
                success=True,
                message="Redis Docker container started successfully",
                details={
                    "method": "docker",
                    "container_name": "redis-raptor",
                    "port": self.redis_port,
                    "password_protected": True
                },
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Redis Docker setup failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def install_redis_portable(self) -> DeploymentResult:
        """Install portable Redis binary"""
        start_time = time.time()
        
        try:
            logger.info("📦 Installing portable Redis...")
            
            # Create local Redis directory
            self.redis_dir.mkdir(exist_ok=True)
            
            system = platform.system().lower()
            machine = platform.machine().lower()
            
            # Redis download URLs (latest stable)
            redis_urls = {
                "linux": {
                    "x86_64": "https://download.redis.io/redis-stable/src/redis-stable.tar.gz",
                    "aarch64": "https://download.redis.io/redis-stable/src/redis-stable.tar.gz"
                }
            }
            
            if system not in redis_urls:
                return DeploymentResult(
                    success=False,
                    message=f"Portable Redis not available for {system}",
                    details={"system": system, "machine": machine},
                    duration_seconds=time.time() - start_time
                )
            
            # For Linux, we'll compile from source
            if system == "linux":
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Download Redis source
                    logger.info("📥 Downloading Redis source...")
                    redis_tar = temp_path / "redis.tar.gz"
                    urllib.request.urlretrieve(redis_urls[system]["x86_64"], redis_tar)
                    
                    # Extract
                    logger.info("📦 Extracting Redis source...")
                    with tarfile.open(redis_tar, 'r:gz') as tar:
                        tar.extractall(temp_path)
                    
                    # Find extracted directory
                    redis_src_dir = next(temp_path.glob("redis-*"))
                    
                    # Compile Redis
                    logger.info("🔨 Compiling Redis (this may take a few minutes)...")
                    compile_result = subprocess.run(
                        ["make", "-j", str(os.cpu_count() or 1)],
                        cwd=redis_src_dir,
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minutes timeout
                    )
                    
                    if compile_result.returncode != 0:
                        return DeploymentResult(
                            success=False,
                            message="Redis compilation failed",
                            details={"error": compile_result.stderr},
                            duration_seconds=time.time() - start_time
                        )
                    
                    # Copy binaries
                    src_dir = redis_src_dir / "src"
                    for binary in ["redis-server", "redis-cli"]:
                        src_binary = src_dir / binary
                        dst_binary = self.redis_dir / binary
                        if src_binary.exists():
                            shutil.copy2(src_binary, dst_binary)
                            dst_binary.chmod(0o755)
                    
                    logger.info("✅ Redis compiled and installed")
            
            return DeploymentResult(
                success=True,
                message="Portable Redis installed successfully",
                details={
                    "method": "portable",
                    "redis_dir": str(self.redis_dir),
                    "binaries": list(self.redis_dir.glob("redis-*"))
                },
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Portable Redis installation failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def start_redis_local(self) -> DeploymentResult:
        """Start local Redis server"""
        start_time = time.time()
        
        try:
            # Find Redis binary
            redis_binary = None
            
            # Check system Redis
            if shutil.which("redis-server"):
                redis_binary = "redis-server"
            # Check local Redis
            elif (self.redis_dir / "redis-server").exists():
                redis_binary = str(self.redis_dir / "redis-server")
            else:
                return DeploymentResult(
                    success=False,
                    message="Redis binary not found",
                    details={"searched_paths": ["/usr/bin/redis-server", str(self.redis_dir / "redis-server")]},
                    duration_seconds=time.time() - start_time
                )
            
            # Create Redis configuration
            redis_config = f"""
# RAPTOR Production Redis Configuration
port {self.redis_port}
bind 127.0.0.1
requirepass {self.redis_password}
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
dir {self.redis_dir.absolute()}
logfile {self.redis_dir.absolute()}/redis.log
loglevel notice
"""
            
            config_path = Path(self.config_file)
            with open(config_path, 'w') as f:
                f.write(redis_config)
            
            # Start Redis server
            logger.info(f"🚀 Starting Redis server: {redis_binary}")
            
            self.redis_process = subprocess.Popen(
                [redis_binary, str(config_path.absolute())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.redis_dir if self.redis_dir.exists() else "."
            )
            
            # Wait for Redis to start
            time.sleep(3)
            
            # Test connection
            if self.redis_process.poll() is None:  # Process still running
                try:
                    import redis
                    r = redis.Redis(
                        host='localhost',
                        port=self.redis_port,
                        password=self.redis_password,
                        socket_connect_timeout=5
                    )
                    r.ping()
                    
                    return DeploymentResult(
                        success=True,
                        message=f"Redis server started successfully (PID: {self.redis_process.pid})",
                        details={
                            "method": "local",
                            "binary": redis_binary,
                            "config_file": str(config_path.absolute()),
                            "pid": self.redis_process.pid,
                            "port": self.redis_port
                        },
                        duration_seconds=time.time() - start_time
                    )
                    
                except Exception as e:
                    return DeploymentResult(
                        success=False,
                        message=f"Redis started but connection test failed: {str(e)}",
                        details={"connection_error": str(e)},
                        duration_seconds=time.time() - start_time
                    )
            else:
                stdout, stderr = self.redis_process.communicate()
                return DeploymentResult(
                    success=False,
                    message="Redis process died immediately",
                    details={
                        "stdout": stdout.decode() if stdout else "",
                        "stderr": stderr.decode() if stderr else "",
                        "return_code": self.redis_process.returncode
                    },
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Redis local startup failed: {str(e)}",
                details={"error": str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def auto_setup_redis(self) -> DeploymentResult:
        """Automatically setup Redis using the best available method"""
        logger.info("🔍 Auto-detecting Redis setup method...")
        
        # Check what's available
        redis_available, method, details = self.check_redis_available()
        
        if redis_available and method == "system":
            logger.info("✅ Redis already running on system")
            return DeploymentResult(
                success=True,
                message="Redis already available and running",
                details=details,
                duration_seconds=0
            )
        
        # Try methods in order of preference
        methods_to_try = [
            ("docker", self.setup_redis_docker, "🐳 Trying Docker Redis..."),
            ("system_install", self.install_redis_system, "📦 Trying system package installation..."),
            ("portable", self.install_redis_portable, "🔧 Trying portable Redis installation...")
        ]
        
        last_error = None
        
        for method_name, method_func, description in methods_to_try:
            try:
                logger.info(description)
                result = method_func()
                
                if result.success:
                    # If installation succeeded, try to start Redis
                    if method_name == "portable":
                        start_result = self.start_redis_local()
                        if start_result.success:
                            return start_result
                        else:
                            last_error = start_result
                    else:
                        return result
                else:
                    last_error = result
                    logger.warning(f"⚠️ {method_name} failed: {result.message}")
                    
            except Exception as e:
                last_error = DeploymentResult(
                    success=False,
                    message=f"{method_name} failed with exception: {str(e)}",
                    details={"error": str(e)},
                    duration_seconds=0
                )
                logger.warning(f"⚠️ {method_name} failed: {e}")
        
        # All methods failed
        return DeploymentResult(
            success=False,
            message="All Redis setup methods failed",
            details={"last_error": last_error.to_dict() if last_error else {}},
            duration_seconds=0
        )
    
    def stop_redis(self):
        """Stop Redis server"""
        if self.redis_process:
            logger.info("🛑 Stopping Redis server...")
            self.redis_process.terminate()
            try:
                self.redis_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.redis_process.kill()
                self.redis_process.wait()
            logger.info("✅ Redis server stopped")

class EnhancedServiceManager:
    """Enhanced service manager with Redis auto-setup"""
    
    def __init__(self, config_env: str = "production"):
        try:
            from production_config import get_production_config
            self.config = get_production_config(config_env)
        except ImportError:
            # Fallback minimal config
            from dataclasses import dataclass
            @dataclass
            class MinimalConfig:
                server_host: str = "0.0.0.0"
                server_port: int = 8000
                workers: int = 1
                worker_timeout: int = 120
            
            self.config = MinimalConfig()
        
        self.redis_manager = RedisManager()
        self.process: Optional[subprocess.Popen] = None
        self.monitoring = None
    
    def setup_redis(self) -> DeploymentResult:
        """Setup Redis with automatic method detection"""
        logger.info("🔧 Setting up Redis...")
        return self.redis_manager.auto_setup_redis()
    
    def start_raptor_server(self, script_path: str = "generic-qa-server.py") -> DeploymentResult:
        """Start RAPTOR server with enhanced configuration"""
        start_time = time.time()
        
        try:
            if not os.path.exists(script_path):
                return DeploymentResult(
                    success=False,
                    message=f"RAPTOR server script not found: {script_path}",
                    details={'script_path': script_path},
                    duration_seconds=time.time() - start_time
                )
            
            # Prepare environment
            env = os.environ.copy()
            env.update({
                'RAPTOR_ENV': 'production',
                'SERVER_HOST': getattr(self.config, 'server_host', '0.0.0.0'),
                'SERVER_PORT': str(getattr(self.config, 'server_port', 8000)),
                'REDIS_HOST': 'localhost',
                'REDIS_PORT': str(self.redis_manager.redis_port),
                'REDIS_PASSWORD': self.redis_manager.redis_password,
            })
            
            # Start server
            logger.info(f"🚀 Starting RAPTOR server: {script_path}")
            
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for startup
            time.sleep(8)
            
            # Check if process is running
            if self.process.poll() is None:
                return DeploymentResult(
                    success=True,
                    message=f"RAPTOR server started (PID: {self.process.pid})",
                    details={
                        'pid': self.process.pid,
                        'host': getattr(self.config, 'server_host', '0.0.0.0'),
                        'port': getattr(self.config, 'server_port', 8000),
                        'url': f"http://{getattr(self.config, 'server_host', '0.0.0.0')}:{getattr(self.config, 'server_port', 8000)}"
                    },
                    duration_seconds=time.time() - start_time
                )
            else:
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
            
            # Stop Redis
            self.redis_manager.stop_redis()
            stopped_services.append("redis")
            
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

class EnhancedProductionDeployer:
    """Enhanced production deployer with automatic Redis setup"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.service_manager = EnhancedServiceManager(environment)
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
    
    def validate_basic_requirements(self) -> DeploymentResult:
        """Validate basic requirements"""
        start_time = time.time()
        
        try:
            checks = {}
            
            # OpenAI API key
            checks['openai_key'] = bool(os.getenv('OPENAI_API_KEY'))
            
            # RAPTOR tree
            tree_path = "vectordb/raptor-production"
            checks['raptor_tree'] = os.path.exists(tree_path)
            
            # Python packages
            required_packages = ['openai', 'fastapi', 'uvicorn']
            package_status = {}
            for package in required_packages:
                try:
                    __import__(package)
                    package_status[package] = True
                except ImportError:
                    package_status[package] = False
            
            checks['packages'] = all(package_status.values())
            
            # Memory (minimum 4GB)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            checks['memory'] = memory_gb >= 4.0
            
            all_passed = all(checks.values())
            
            return DeploymentResult(
                success=all_passed,
                message="Basic validation passed" if all_passed else "Basic validation failed",
                details={
                    'checks': checks,
                    'package_status': package_status,
                    'memory_gb': round(memory_gb, 2)
                },
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Validation error: {str(e)}",
                details={'error': str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def deploy(self, skip_validation: bool = False) -> DeploymentResult:
        """Enhanced deployment with automatic Redis setup"""
        deployment_start = time.time()
        
        logger.info("🚀 Enhanced RAPTOR deployment with automatic Redis setup")
        logger.info("="*60)
        
        try:
            # Step 1: Basic validation
            if not skip_validation:
                logger.info("🔍 Step 1: Basic validation...")
                validation_result = self.validate_basic_requirements()
                self.log_step("Basic Validation", validation_result)
                
                if not validation_result.success:
                    return DeploymentResult(
                        success=False,
                        message="Basic validation failed",
                        details={'validation': validation_result.details},
                        duration_seconds=time.time() - deployment_start
                    )
            
            # Step 2: Automatic Redis setup
            logger.info("🔧 Step 2: Automatic Redis setup...")
            redis_result = self.service_manager.setup_redis()
            self.log_step("Redis Setup", redis_result)
            
            if not redis_result.success:
                return DeploymentResult(
                    success=False,
                    message="Redis setup failed",
                    details={'redis': redis_result.details},
                    duration_seconds=time.time() - deployment_start
                )
            
            # Step 3: Start RAPTOR server
            logger.info("🚀 Step 3: Starting RAPTOR server...")
            server_result = self.service_manager.start_raptor_server()
            self.log_step("RAPTOR Server Startup", server_result)
            
            if not server_result.success:
                return DeploymentResult(
                    success=False,
                    message="RAPTOR server startup failed",
                    details={'server': server_result.details},
                    duration_seconds=time.time() - deployment_start
                )
            
            # Step 4: Health check
            logger.info("🔍 Step 4: Health check...")
            time.sleep(5)
            
            try:
                import requests
                health_url = server_result.details.get('url', 'http://localhost:8000') + '/health'
                response = requests.get(health_url, timeout=10)
                health_status = response.json() if response.status_code == 200 else {"status": "unhealthy"}
            except:
                health_status = {"status": "unknown"}
            
            # Success!
            total_time = time.time() - deployment_start
            success_message = f"🎉 Enhanced RAPTOR deployment successful! ({total_time:.1f}s)"
            logger.info(success_message)
            
            return DeploymentResult(
                success=True,
                message="Enhanced deployment completed successfully",
                details={
                    'server_url': server_result.details.get('url', 'http://localhost:8000'),
                    'health_check_url': health_url,
                    'redis_method': redis_result.details.get('method', 'unknown'),
                    'health_status': health_status,
                    'deployment_log': self.deployment_log
                },
                duration_seconds=total_time
            )
            
        except Exception as e:
            logger.error(f"💥 Deployment failed: {str(e)}")
            
            # Cleanup
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
    
    def rollback(self) -> DeploymentResult:
        """Rollback deployment"""
        logger.info("⏪ Rolling back deployment...")
        return self.service_manager.stop_services()

def main():
    """Main deployment entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RAPTOR Production Deployer with Auto Redis")
    parser.add_argument("--env", choices=["development", "staging", "production"], 
                       default="production", help="Deployment environment")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip system validation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate but don't deploy")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = EnhancedProductionDeployer(args.env)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        deployer.rollback()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.dry_run:
        logger.info("🧪 Dry run - validation only")
        result = deployer.validate_basic_requirements()
        
        print("\n" + "="*60)
        print("🧪 DRY RUN RESULTS")
        print("="*60)
        print(f"Validation: {'✅ PASS' if result.success else '❌ FAIL'}")
        if not result.success:
            print(f"Issues: {result.message}")
        print("="*60)
        sys.exit(0 if result.success else 1)
    
    # Run deployment
    result = deployer.deploy(args.skip_validation)
    
    if result.success:
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT SUCCESSFUL")
        print("="*60)
        print(f"🌐 Server: {result.details.get('server_url')}")
        print(f"🔍 Health: {result.details.get('health_check_url')}")
        print(f"🔧 Redis Method: {result.details.get('redis_method')}")
        print("="*60)
        
        # Keep running
        try:
            logger.info("🔄 Deployment running... Press Ctrl+C to stop")
            while True:
                time.sleep(30)
                try:
                    import requests
                    response = requests.get(result.details.get('health_check_url'), timeout=5)
                    status = response.json().get('status', 'unknown') if response.status_code == 200 else 'unhealthy'
                    logger.info(f"Health: {status}")
                except:
                    logger.warning("Health check failed")
                    
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