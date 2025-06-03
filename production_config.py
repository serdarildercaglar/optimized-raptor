"""
DOSYA: production-config.py  
AÇIKLAMA: Production environment configuration management - Environment-based settings
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    SPEED = "speed"           # Maximum throughput
    BALANCED = "balanced"     # Balance between speed and quality
    QUALITY = "quality"       # Maximum quality
    MEMORY_OPTIMIZED = "memory"  # Low memory usage

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    connection_pool_size: int = 10
    connection_timeout: int = 30

@dataclass
class RAPTORConfig:
    """RAPTOR-specific configuration"""
    # Tree building parameters
    max_tokens: int = 100
    summarization_length: int = 400
    num_layers: int = 5
    batch_size: int = 100
    
    # Retrieval parameters
    top_k: int = 8
    threshold: float = 0.5
    
    # Performance parameters
    enable_caching: bool = True
    enable_async: bool = True
    enable_metrics: bool = True
    max_concurrent_operations: int = 10
    cache_ttl: int = 3600
    
    # Quality parameters
    adaptive_retrieval: bool = True
    early_termination: bool = True
    confidence_threshold: float = 0.8

@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: Optional[str] = None
    model_summarization: str = "gpt-4o"
    model_qa: str = "gpt-4.1"
    max_retries: int = 6
    timeout: int = 60
    rate_limit_rpm: int = 500  # Requests per minute

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval: int = 30
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    metrics_retention_days: int = 30

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_cors: bool = True
    allowed_origins: list = None
    rate_limit_per_minute: int = 100
    max_request_size: int = 50_000_000  # 50MB
    enable_request_logging: bool = True
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]

@dataclass
class ProductionConfig:
    """Main production configuration container"""
    environment: Environment
    raptor: RAPTORConfig
    database: DatabaseConfig
    openai: OpenAIConfig
    monitoring: MonitoringConfig
    security: SecurityConfig
    
    # Server configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    workers: int = 1
    worker_timeout: int = 120
    
    # Data paths
    data_path: str = "data"
    vectordb_path: str = "vectordb"
    logs_path: str = "logs"
    metrics_path: str = "metrics"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'environment': self.environment.value,
            'raptor': asdict(self.raptor),
            'database': asdict(self.database),
            'openai': asdict(self.openai),
            'monitoring': asdict(self.monitoring),
            'security': asdict(self.security),
            'server_host': self.server_host,
            'server_port': self.server_port,
            'workers': self.workers,
            'worker_timeout': self.worker_timeout,
            'data_path': self.data_path,
            'vectordb_path': self.vectordb_path,
            'logs_path': self.logs_path,
            'metrics_path': self.metrics_path
        }

class ConfigManager:
    """Configuration manager with environment-specific settings"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def get_config(self, 
                   environment: Environment = None,
                   performance_profile: PerformanceProfile = None) -> ProductionConfig:
        """
        Get configuration for specified environment and performance profile
        
        Args:
            environment: Target environment (auto-detect if None)
            performance_profile: Performance profile (balanced if None)
        
        Returns:
            Production configuration
        """
        
        # Auto-detect environment if not specified
        if environment is None:
            env_name = os.getenv('RAPTOR_ENV', 'development').lower()
            environment = Environment(env_name)
        
        # Default performance profile
        if performance_profile is None:
            profile_name = os.getenv('RAPTOR_PROFILE', 'balanced').lower()
            performance_profile = PerformanceProfile(profile_name)
        
        logger.info(f"Loading configuration for {environment.value} environment with {performance_profile.value} profile")
        
        # Base configuration
        config = self._get_base_config(environment)
        
        # Apply performance profile
        config = self._apply_performance_profile(config, performance_profile)
        
        # Apply environment-specific overrides
        config = self._apply_environment_overrides(config, environment)
        
        # Apply environment variables
        config = self._apply_env_variables(config)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _get_base_config(self, environment: Environment) -> ProductionConfig:
        """Get base configuration for environment"""
        
        if environment == Environment.DEVELOPMENT:
            return ProductionConfig(
                environment=environment,
                raptor=RAPTORConfig(
                    max_tokens=100,
                    num_layers=3,
                    batch_size=50,
                    max_concurrent_operations=4,
                    enable_metrics=False  # Disable for dev
                ),
                database=DatabaseConfig(
                    redis_password=None,  # No password for dev
                    connection_pool_size=5
                ),
                openai=OpenAIConfig(
                    rate_limit_rpm=100  # Lower for dev
                ),
                monitoring=MonitoringConfig(
                    enable_prometheus=False,  # Disable for dev
                    log_level="DEBUG"
                ),
                security=SecurityConfig(
                    rate_limit_per_minute=1000,  # Higher for dev
                    enable_request_logging=False
                ),
                workers=1,
                server_port=8000
            )
        
        elif environment == Environment.STAGING:
            return ProductionConfig(
                environment=environment,
                raptor=RAPTORConfig(
                    max_tokens=100,
                    num_layers=4,
                    batch_size=75,
                    max_concurrent_operations=8
                ),
                database=DatabaseConfig(
                    connection_pool_size=10
                ),
                openai=OpenAIConfig(
                    rate_limit_rpm=300
                ),
                monitoring=MonitoringConfig(
                    enable_prometheus=True,
                    log_level="INFO"
                ),
                security=SecurityConfig(
                    rate_limit_per_minute=200
                ),
                workers=2,
                server_port=8000
            )
        
        else:  # PRODUCTION
            return ProductionConfig(
                environment=environment,
                raptor=RAPTORConfig(
                    max_tokens=100,
                    num_layers=5,
                    batch_size=150,
                    max_concurrent_operations=12,
                    cache_ttl=7200  # 2 hours for production
                ),
                database=DatabaseConfig(
                    connection_pool_size=20,
                    connection_timeout=30
                ),
                openai=OpenAIConfig(
                    rate_limit_rpm=500,
                    max_retries=6
                ),
                monitoring=MonitoringConfig(
                    enable_prometheus=True,
                    log_level="INFO",
                    metrics_retention_days=90
                ),
                security=SecurityConfig(
                    rate_limit_per_minute=100,
                    allowed_origins=["https://yourdomain.com"],
                    enable_request_logging=True
                ),
                workers=4,
                server_port=8000
            )
    
    def _apply_performance_profile(self, 
                                 config: ProductionConfig, 
                                 profile: PerformanceProfile) -> ProductionConfig:
        """Apply performance profile optimizations"""
        
        if profile == PerformanceProfile.SPEED:
            # Optimize for maximum throughput
            config.raptor.batch_size = int(config.raptor.batch_size * 1.5)
            config.raptor.max_concurrent_operations = min(config.raptor.max_concurrent_operations + 4, 16)
            config.raptor.cache_ttl = 7200  # 2 hours
            config.raptor.early_termination = True
            config.raptor.confidence_threshold = 0.7  # Lower threshold for speed
            
        elif profile == PerformanceProfile.QUALITY:
            # Optimize for maximum quality
            config.raptor.summarization_length = int(config.raptor.summarization_length * 1.5)
            config.raptor.num_layers = min(config.raptor.num_layers + 1, 6)
            config.raptor.top_k = 10
            config.raptor.threshold = 0.4
            config.raptor.confidence_threshold = 0.9  # Higher threshold for quality
            
        elif profile == PerformanceProfile.MEMORY_OPTIMIZED:
            # Optimize for low memory usage
            config.raptor.batch_size = max(config.raptor.batch_size // 2, 25)
            config.raptor.max_concurrent_operations = max(config.raptor.max_concurrent_operations // 2, 4)
            config.raptor.cache_ttl = 1800  # 30 minutes
            config.workers = max(config.workers // 2, 1)
            
        # BALANCED uses default settings
        
        return config
    
    def _apply_environment_overrides(self, 
                                   config: ProductionConfig, 
                                   environment: Environment) -> ProductionConfig:
        """Apply environment-specific overrides"""
        
        # Load environment-specific config file if exists
        config_file = self.config_dir / f"{environment.value}.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    overrides = json.load(f)
                
                # Apply overrides (basic implementation)
                for section, values in overrides.items():
                    if hasattr(config, section):
                        section_obj = getattr(config, section)
                        if hasattr(section_obj, '__dict__'):
                            for key, value in values.items():
                                if hasattr(section_obj, key):
                                    setattr(section_obj, key, value)
                        else:
                            setattr(config, section, value)
                
                logger.info(f"Applied overrides from {config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load config overrides: {e}")
        
        return config
    
    def _apply_env_variables(self, config: ProductionConfig) -> ProductionConfig:
        """Apply environment variable overrides"""
        
        # OpenAI API Key (required)
        if api_key := os.getenv('OPENAI_API_KEY'):
            config.openai.api_key = api_key
        
        # Redis configuration
        if redis_host := os.getenv('REDIS_HOST'):
            config.database.redis_host = redis_host
        
        if redis_port := os.getenv('REDIS_PORT'):
            config.database.redis_port = int(redis_port)
        
        if redis_password := os.getenv('REDIS_PASSWORD'):
            config.database.redis_password = redis_password
        
        # Server configuration
        if server_host := os.getenv('SERVER_HOST'):
            config.server_host = server_host
        
        if server_port := os.getenv('SERVER_PORT'):
            config.server_port = int(server_port)
        
        # Paths
        if data_path := os.getenv('DATA_PATH'):
            config.data_path = data_path
        
        if vectordb_path := os.getenv('VECTORDB_PATH'):
            config.vectordb_path = vectordb_path
        
        # Performance tuning
        if max_workers := os.getenv('MAX_WORKERS'):
            config.workers = int(max_workers)
        
        if batch_size := os.getenv('RAPTOR_BATCH_SIZE'):
            config.raptor.batch_size = int(batch_size)
        
        return config
    
    def _validate_config(self, config: ProductionConfig):
        """Validate configuration"""
        
        errors = []
        
        # Required fields
        if not config.openai.api_key:
            errors.append("OpenAI API key is required")
        
        # Reasonable limits
        if config.raptor.batch_size < 10 or config.raptor.batch_size > 500:
            errors.append(f"Batch size {config.raptor.batch_size} is outside reasonable range (10-500)")
        
        if config.raptor.max_concurrent_operations > 20:
            errors.append(f"Max concurrent operations {config.raptor.max_concurrent_operations} may cause resource issues")
        
        # Path validation
        for path_name in ['data_path', 'vectordb_path', 'logs_path', 'metrics_path']:
            path_value = getattr(config, path_name)
            try:
                Path(path_value).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {path_value}: {e}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        logger.info("✅ Configuration validation passed")
    
    def save_config(self, config: ProductionConfig, filename: str = None):
        """Save configuration to file"""
        
        if filename is None:
            filename = f"{config.environment.value}_config.json"
        
        config_path = self.config_dir / filename
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")

# Global config manager instance
config_manager = ConfigManager()

def get_production_config(environment: str = None, 
                         profile: str = None) -> ProductionConfig:
    """
    Convenience function to get production configuration
    
    Args:
        environment: Environment name (development/staging/production)
        profile: Performance profile (speed/balanced/quality/memory)
    
    Returns:
        Production configuration
    """
    
    env = Environment(environment) if environment else None
    perf_profile = PerformanceProfile(profile) if profile else None
    
    return config_manager.get_config(env, perf_profile)

def create_sample_configs():
    """Create sample configuration files for each environment"""
    
    # Development overrides
    dev_overrides = {
        "raptor": {
            "enable_metrics": False,
            "batch_size": 25
        },
        "monitoring": {
            "enable_prometheus": False,
            "log_level": "DEBUG"
        }
    }
    
    # Staging overrides
    staging_overrides = {
        "raptor": {
            "num_layers": 4,
            "batch_size": 75
        },
        "security": {
            "rate_limit_per_minute": 200
        }
    }
    
    # Production overrides
    prod_overrides = {
        "raptor": {
            "cache_ttl": 7200,
            "batch_size": 200
        },
        "security": {
            "allowed_origins": ["https://yourdomain.com"],
            "rate_limit_per_minute": 100
        }
    }
    
    # Save sample configs
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    for env_name, overrides in [
        ("development", dev_overrides),
        ("staging", staging_overrides), 
        ("production", prod_overrides)
    ]:
        config_path = config_dir / f"{env_name}.json"
        with open(config_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        
        print(f"Created sample config: {config_path}")

if __name__ == "__main__":
    # Create sample configuration files
    create_sample_configs()
    
    # Test configuration loading
    for env in ["development", "staging", "production"]:
        for profile in ["speed", "balanced", "quality"]:
            print(f"\n=== Testing {env} + {profile} ===")
            try:
                config = get_production_config(env, profile)
                print(f"✅ {env}/{profile}: {config.raptor.batch_size} batch size, {config.workers} workers")
            except Exception as e:
                print(f"❌ {env}/{profile}: {e}")