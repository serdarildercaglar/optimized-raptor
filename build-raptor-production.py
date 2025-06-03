"""
DOSYA: build-raptor-production.py
AÇIKLAMA: Enterprise-level RAPTOR build script - Production ortamı için optimize edilmiş
"""

import os
import sys
import time
import psutil
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from contextlib import contextmanager

# Environment management
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# RAPTOR imports
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import GPT41SummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('raptor_build.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProductionMetrics:
    """Production build metrics tracking"""
    start_time: float
    end_time: Optional[float] = None
    total_documents: int = 0
    total_chunks: int = 0
    total_nodes: int = 0
    layers_built: int = 0
    memory_peak_mb: float = 0.0
    cpu_peak_percent: float = 0.0
    cache_hit_rate: float = 0.0
    embedding_efficiency: float = 0.0
    build_errors: List[str] = None
    
    def __post_init__(self):
        if self.build_errors is None:
            self.build_errors = []
    
    @property
    def total_time(self) -> float:
        return (self.end_time or time.time()) - self.start_time
    
    @property
    def throughput_chunks_per_sec(self) -> float:
        return self.total_chunks / max(self.total_time, 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_time_seconds': self.total_time,
            'total_documents': self.total_documents,
            'total_chunks': self.total_chunks,
            'total_nodes': self.total_nodes,
            'layers_built': self.layers_built,
            'memory_peak_mb': self.memory_peak_mb,
            'cpu_peak_percent': self.cpu_peak_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'embedding_efficiency': self.embedding_efficiency,
            'throughput_chunks_per_sec': self.throughput_chunks_per_sec,
            'build_errors': self.build_errors,
            'timestamp': datetime.now().isoformat()
        }

class SystemMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.peak_memory = 0.0
        self.peak_cpu = 0.0
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("🔍 System monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info(f"📊 Peak Memory: {self.peak_memory:.1f}MB, Peak CPU: {self.peak_cpu:.1f}%")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.peak_cpu = max(self.peak_cpu, cpu_percent)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                
            time.sleep(5)  # Monitor every 5 seconds

class ProductionValidator:
    """Production environment validation"""
    
    @staticmethod
    def validate_environment() -> Dict[str, bool]:
        """Validate production environment requirements"""
        checks = {}
        
        # OpenAI API key check
        checks['openai_api_key'] = bool(os.getenv('OPENAI_API_KEY'))
        
        # Python version check
        checks['python_version'] = sys.version_info >= (3, 8)
        
        # Memory check (minimum 4GB)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        checks['memory_sufficient'] = memory_gb >= 4.0
        
        # Disk space check (minimum 10GB free)
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        checks['disk_space'] = free_gb >= 10.0
        
        # Required packages
        try:
            import torch
            import transformers
            import openai
            checks['required_packages'] = True
        except ImportError:
            checks['required_packages'] = False
        
        return checks
    
    @staticmethod
    def validate_input_data(data_path: str) -> Dict[str, Any]:
        """Validate input data"""
        validation = {
            'exists': False,
            'readable': False,
            'size_mb': 0.0,
            'estimated_chunks': 0,
            'recommended_config': {}
        }
        
        if not os.path.exists(data_path):
            return validation
        
        validation['exists'] = True
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            validation['readable'] = True
            validation['size_mb'] = len(content.encode('utf-8')) / (1024 * 1024)
            
            # Estimate chunks (rough calculation)
            avg_words_per_chunk = 75  # ~100 tokens
            words = len(content.split())
            validation['estimated_chunks'] = max(1, words // avg_words_per_chunk)
            
            # Recommend configuration based on size
            if validation['size_mb'] < 1:  # Small
                validation['recommended_config'] = {
                    'chunk_size': 150,
                    'num_layers': 3,
                    'batch_size': 50
                }
            elif validation['size_mb'] < 10:  # Medium
                validation['recommended_config'] = {
                    'chunk_size': 120,
                    'num_layers': 4,
                    'batch_size': 100
                }
            else:  # Large
                validation['recommended_config'] = {
                    'chunk_size': 100,
                    'num_layers': 5,
                    'batch_size': 150
                }
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
        
        return validation

@contextmanager
def production_build_context(metrics: ProductionMetrics):
    """Context manager for production build with cleanup"""
    monitor = SystemMonitor()
    
    try:
        logger.info("🚀 Starting production RAPTOR build")
        monitor.start_monitoring()
        yield monitor
        
    except Exception as e:
        metrics.build_errors.append(str(e))
        logger.error(f"Build failed: {e}")
        raise
        
    finally:
        monitor.stop_monitoring()
        metrics.memory_peak_mb = monitor.peak_memory
        metrics.cpu_peak_percent = monitor.peak_cpu
        metrics.end_time = time.time()
        
        logger.info(f"✅ Build completed in {metrics.total_time:.1f}s")

def create_production_config(
    document_size_mb: float,
    performance_profile: str = "balanced"
) -> RetrievalAugmentationConfig:
    """
    Create production-optimized configuration
    
    Args:
        document_size_mb: Document size in MB
        performance_profile: "speed", "balanced", "quality"
    """
    
    # Base configuration based on document size
    if document_size_mb < 1:  # Small documents
        base_config = {
            'tb_max_tokens': 150,
            'tb_summarization_length': 200,
            'tb_num_layers': 3,
            'tb_batch_size': 50,
            'max_concurrent_operations': 8
        }
    elif document_size_mb < 10:  # Medium documents
        base_config = {
            'tb_max_tokens': 120,
            'tb_summarization_length': 400,
            'tb_num_layers': 4,
            'tb_batch_size': 100,
            'max_concurrent_operations': 10
        }
    else:  # Large documents
        base_config = {
            'tb_max_tokens': 100,
            'tb_summarization_length': 512,
            'tb_num_layers': 5,
            'tb_batch_size': 150,
            'max_concurrent_operations': 12
        }
    
    # Adjust based on performance profile
    if performance_profile == "speed":
        base_config.update({
            'tb_batch_size': int(base_config['tb_batch_size'] * 1.5),
            'max_concurrent_operations': min(base_config['max_concurrent_operations'] + 4, 16),
            'cache_ttl': 7200,  # 2 hours
            'tr_early_termination': True
        })
    elif performance_profile == "quality":
        base_config.update({
            'tb_summarization_length': int(base_config['tb_summarization_length'] * 1.5),
            'tb_num_layers': base_config['tb_num_layers'] + 1,
            'tr_top_k': 10,
            'tr_threshold': 0.4
        })
    # "balanced" uses base config
    
    # Initialize models
    embed_model = CustomEmbeddingModel()
    sum_model = GPT4OSummarizationModel()
    
    return RetrievalAugmentationConfig(
        # Tree building
        tb_max_tokens=base_config['tb_max_tokens'],
        tb_summarization_length=base_config['tb_summarization_length'],
        tb_num_layers=base_config['tb_num_layers'],
        tb_batch_size=base_config['tb_batch_size'],
        tb_build_mode="async",
        tb_enable_progress_tracking=True,
        
        # Retrieval optimization
        tr_enable_caching=True,
        tr_adaptive_retrieval=True,
        tr_early_termination=base_config.get('tr_early_termination', True),
        tr_top_k=base_config.get('tr_top_k', 8),
        tr_threshold=base_config.get('tr_threshold', 0.5),
        
        # Performance
        enable_async=True,
        enable_metrics=True,
        enable_progress_tracking=True,
        performance_monitoring=True,
        max_concurrent_operations=base_config['max_concurrent_operations'],
        cache_ttl=base_config.get('cache_ttl', 3600),
        
        # Models
        summarization_model=sum_model,
        embedding_model=embed_model,
        tree_builder_type="cluster"
    )

def production_progress_callback(progress, metrics: ProductionMetrics):
    """Enhanced progress callback with metrics collection"""
    metrics.total_chunks = max(metrics.total_chunks, progress.total_chunks)
    metrics.total_nodes = progress.created_nodes
    metrics.layers_built = progress.current_layer
    
    # Log progress every 10 chunks or layer change
    if progress.processed_chunks % 10 == 0 or progress.current_layer != getattr(production_progress_callback, 'last_layer', 0):
        logger.info(
            f"📊 Layer {progress.current_layer}/{progress.total_layers} | "
            f"Chunks: {progress.processed_chunks}/{progress.total_chunks} | "
            f"Nodes: {progress.created_nodes} | "
            f"Time: {progress.elapsed_time:.1f}s"
        )
        production_progress_callback.last_layer = progress.current_layer

def build_raptor_production(
    data_path: str,
    output_path: str = "vectordb/raptor-production",
    performance_profile: str = "balanced",
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    Production RAPTOR build with comprehensive validation and monitoring
    
    Args:
        data_path: Path to input data file
        output_path: Path to save the built tree
        performance_profile: "speed", "balanced", or "quality"
        force_rebuild: Force rebuild even if output exists
    
    Returns:
        Build results and metrics
    """
    
    # Initialize metrics
    metrics = ProductionMetrics(start_time=time.time())
    
    try:
        # Pre-build validation
        logger.info("🔍 Running pre-build validation...")
        
        # Environment validation
        env_checks = ProductionValidator.validate_environment()
        failed_checks = [k for k, v in env_checks.items() if not v]
        
        if failed_checks:
            raise RuntimeError(f"Environment validation failed: {failed_checks}")
        
        logger.info("✅ Environment validation passed")
        
        # Data validation
        data_validation = ProductionValidator.validate_input_data(data_path)
        
        if not data_validation['exists']:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if not data_validation['readable']:
            raise RuntimeError(f"Data file not readable: {data_path}")
        
        logger.info(
            f"📄 Data validated: {data_validation['size_mb']:.1f}MB, "
            f"~{data_validation['estimated_chunks']} chunks"
        )
        
        # Check if output already exists
        if os.path.exists(output_path) and not force_rebuild:
            logger.warning(f"Output already exists: {output_path} (use force_rebuild=True to overwrite)")
            return {
                'status': 'skipped',
                'reason': 'output_exists',
                'output_path': output_path
            }
        
        # Load and validate data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        metrics.total_documents = 1
        
        # Create optimized configuration
        config = create_production_config(
            data_validation['size_mb'],
            performance_profile
        )
        
        logger.info(f"⚙️ Configuration: {performance_profile} profile")
        
        # Build with monitoring
        with production_build_context(metrics) as monitor:
            
            # Initialize RAPTOR
            RA = RetrievalAugmentation(config=config)
            
            # Set progress callback
            RA.set_progress_callback(
                lambda progress: production_progress_callback(progress, metrics)
            )
            
            # Build tree
            logger.info("🏗️ Building RAPTOR tree...")
            RA.add_documents(text)
            
            # Collect final metrics
            perf_summary = RA.get_performance_summary()
            
            if 'retriever' in perf_summary:
                metrics.cache_hit_rate = perf_summary['retriever'].get('cache_hit_rate', 0)
            
            if 'tree_builder' in perf_summary:
                metrics.embedding_efficiency = perf_summary['tree_builder'].get('embedding_efficiency', 0)
            
            # Save tree
            logger.info(f"💾 Saving tree to {output_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            RA.save(output_path, include_metadata=True)
            
            # Performance validation
            logger.info("🧪 Running post-build validation...")
            
            # Quick retrieval test
            test_queries = [
                "What is this document about?",
                "Summarize the main points",
                "Key information"
            ]
            
            retrieval_times = []
            for query in test_queries:
                start_time = time.time()
                context = RA.retrieve(query, max_tokens=1000)
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                if len(context) < 100:
                    metrics.build_errors.append(f"Poor retrieval quality for query: {query}")
            
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            
            # Performance thresholds
            if avg_retrieval_time > 5.0:  # 5 seconds threshold
                metrics.build_errors.append(f"Slow retrieval performance: {avg_retrieval_time:.2f}s avg")
            
            logger.info(f"🎯 Average retrieval time: {avg_retrieval_time:.3f}s")
        
        # Save metrics
        metrics_path = f"{output_path}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Final summary
        result = {
            'status': 'success',
            'output_path': output_path,
            'metrics_path': metrics_path,
            'metrics': metrics.to_dict(),
            'performance_profile': performance_profile,
            'avg_retrieval_time': avg_retrieval_time,
            'tree_stats': {
                'nodes': metrics.total_nodes,
                'layers': metrics.layers_built,
                'chunks': metrics.total_chunks
            }
        }
        
        logger.info("🎉 Production build completed successfully!")
        logger.info(f"📊 Final metrics: {metrics.total_time:.1f}s, {metrics.total_nodes} nodes, {metrics.layers_built} layers")
        
        if metrics.build_errors:
            logger.warning(f"⚠️ Build warnings: {len(metrics.build_errors)} issues detected")
            for error in metrics.build_errors:
                logger.warning(f"  - {error}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Production build failed: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'status': 'failed',
            'error': str(e),
            'metrics': metrics.to_dict() if metrics else {}
        }

def main():
    """Main entry point for production build"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAPTOR Production Builder")
    parser.add_argument("data_path", help="Path to input data file")
    parser.add_argument("-o", "--output", default="vectordb/raptor-production", 
                       help="Output path for RAPTOR tree")
    parser.add_argument("-p", "--profile", choices=["speed", "balanced", "quality"], 
                       default="balanced", help="Performance profile")
    parser.add_argument("--force", action="store_true", 
                       help="Force rebuild if output exists")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run production build
    result = build_raptor_production(
        data_path=args.data_path,
        output_path=args.output,
        performance_profile=args.profile,
        force_rebuild=args.force
    )
    
    # Exit with appropriate code
    if result['status'] == 'success':
        print(f"\n✅ Success! Tree saved to: {result['output_path']}")
        print(f"📊 Metrics saved to: {result['metrics_path']}")
        sys.exit(0)
    elif result['status'] == 'skipped':
        print(f"\n⏭️ Skipped: {result['reason']}")
        sys.exit(0)
    else:
        print(f"\n❌ Failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()