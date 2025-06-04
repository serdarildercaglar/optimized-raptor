"""
DOSYA: generic-qa-server.py
AÇIKLAMA: Enhanced RAPTOR soru-cevap asistanı - Management endpoints ile
"""

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from raptor import RetrievalAugmentation 
from raptor import RetrievalAugmentationConfig
from raptor import GPT41SummarizationModel
from raptor import GPT41QAModel
from raptor.EmbeddingModels import CustomEmbeddingModel

# RAPTOR yapılandırması
embed_model = CustomEmbeddingModel()
sum_model = GPT41SummarizationModel()
qa_model = GPT41QAModel()

RA_config = RetrievalAugmentationConfig(
    tb_summarization_length=100, 
    tb_max_tokens=100, 
    qa_model=qa_model,
    summarization_model=sum_model, 
    embedding_model=embed_model
)

# Önceden oluşturulmuş RAPTOR tree'sini yükle
PATH = "vectordb/raptor-production"
RA = RetrievalAugmentation(tree=PATH, config=RA_config)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import redis.asyncio as redis
import uuid
from openai import AsyncOpenAI
import asyncio
import time
import logging
import psutil
import gc
import os
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI(title="Enhanced RAPTOR QA Server", description="RAPTOR-powered Question Answering System with Management")

# ====================================================================
# ENHANCED LOGGING & MONITORING
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global metrics collector
class SystemMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.websocket_connections = 0
        self.chat_sessions = 0
        self.raptor_queries = 0
        self.last_reset = time.time()
    
    def increment_request(self):
        self.request_count += 1
    
    def increment_error(self):
        self.error_count += 1
    
    def update_connections(self, count: int):
        self.websocket_connections = count
    
    def increment_raptor_query(self):
        self.raptor_queries += 1
    
    def get_uptime_seconds(self):
        return time.time() - self.start_time
    
    def reset_counters(self):
        self.request_count = 0
        self.error_count = 0
        self.raptor_queries = 0
        self.last_reset = time.time()

# Global metrics instance
system_metrics = SystemMetrics()

# ====================================================================
# REDIS & OPENAI CONFIGURATION
# ====================================================================

# Redis yapılandırması
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    password='Ph4nt0m4+4',
    decode_responses=True
)

openai_client = AsyncOpenAI()

# ====================================================================
# ENHANCED CONNECTION MANAGER
# ====================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        system_metrics.chat_sessions += 1  # Track sessions
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def send_stream_chunk(self, chunk: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(chunk))

manager = ConnectionManager()

# ====================================================================
# MIDDLEWARE: Request tracking
# ====================================================================

@app.middleware("http")
async def track_requests(request, call_next):
    """Request tracking middleware"""
    system_metrics.increment_request()
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        system_metrics.increment_error()
        logger.error(f"Request error: {e}")
        raise e

# ====================================================================
# MODEL LOADING STATE TRACKER (Enhanced)
# ====================================================================

MODEL_LOADING_STATE = {
    'models_loaded': False,
    'loading_start_time': time.time(),
    'loading_errors': [],
    'loaded_models': [],
    'embedding_model_ready': False,
    'raptor_tree_ready': False
}

async def check_model_loading_status() -> Dict[str, any]:
    """Model yükleme durumunu kontrol eder"""
    global MODEL_LOADING_STATE
    
    try:
        # RAPTOR embedding model'ini kontrol et
        if hasattr(RA, 'tree') and RA.tree:
            MODEL_LOADING_STATE['raptor_tree_ready'] = True
            if 'raptor_tree' not in MODEL_LOADING_STATE['loaded_models']:
                MODEL_LOADING_STATE['loaded_models'].append('raptor_tree')
        
        # Embedding model'ini kontrol et
        if hasattr(RA, 'tree_retriever') and RA.tree_retriever:
            try:
                # Test embedding creation
                test_embedding = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: RA.tree_retriever.create_embedding("test")
                )
                if test_embedding and len(test_embedding) > 0:
                    MODEL_LOADING_STATE['embedding_model_ready'] = True
                    if 'embedding_model' not in MODEL_LOADING_STATE['loaded_models']:
                        MODEL_LOADING_STATE['loaded_models'].append('embedding_model')
            except Exception as e:
                if str(e) not in MODEL_LOADING_STATE['loading_errors']:
                    MODEL_LOADING_STATE['loading_errors'].append(str(e))
        
        # OpenAI client'ı kontrol et
        try:
            # Basit bir test call (bu genelde hızlıdır)
            test_response = await openai_client.models.list()
            if 'openai_client' not in MODEL_LOADING_STATE['loaded_models']:
                MODEL_LOADING_STATE['loaded_models'].append('openai_client')
        except Exception as e:
            if str(e) not in MODEL_LOADING_STATE['loading_errors']:
                MODEL_LOADING_STATE['loading_errors'].append(str(e))
        
        # Overall model loading status
        required_models = ['raptor_tree', 'embedding_model', 'openai_client']
        loaded_required = [m for m in required_models if m in MODEL_LOADING_STATE['loaded_models']]
        
        MODEL_LOADING_STATE['models_loaded'] = len(loaded_required) >= 2  # En az 2 model yüklü olsun
        
        return MODEL_LOADING_STATE
        
    except Exception as e:
        MODEL_LOADING_STATE['loading_errors'].append(str(e))
        return MODEL_LOADING_STATE

# ====================================================================
# KRITIK 5 MANAGEMENT ENDPOINT
# ====================================================================

# 1. DELETE /cache/clear - Memory leaks prevention
@app.delete("/cache/clear")
async def clear_cache():
    """Tüm cache'leri temizle - Memory leak prevention"""
    try:
        start_time = time.time()
        cleared_items = {
            "redis_cache": 0,
            "raptor_cache": 0,
            "embedding_cache": 0,
            "python_gc": 0
        }
        
        # 1. Redis cache temizle
        try:
            # Chat history dışında cache'leri temizle
            pipeline = redis_client.pipeline()
            
            # Pattern'lere göre cache key'leri bul ve sil
            cache_patterns = [
                "cache:*", 
                "temp:*", 
                "session_temp:*",
                "query_cache:*"
            ]
            
            for pattern in cache_patterns:
                keys = await redis_client.keys(pattern)
                if keys:
                    cleared_items["redis_cache"] += len(keys)
                    for key in keys:
                        pipeline.delete(key)
            
            await pipeline.execute()
            
        except Exception as e:
            logger.warning(f"Redis cache clear error: {e}")
        
        # 2. RAPTOR cache temizle
        try:
            if RA and hasattr(RA, 'clear_all_caches'):
                RA.clear_all_caches()
                cleared_items["raptor_cache"] = 1
            
            if RA and hasattr(RA, 'retriever') and hasattr(RA.retriever, 'clear_cache'):
                RA.retriever.clear_cache()
                cleared_items["raptor_cache"] += 1
                
        except Exception as e:
            logger.warning(f"RAPTOR cache clear error: {e}")
        
        # 3. Embedding model cache temizle
        try:
            if hasattr(RA, 'tree_builder') and hasattr(RA.tree_builder, 'embedding_models'):
                for model_name, model in RA.tree_builder.embedding_models.items():
                    if hasattr(model, 'cache') and model.cache:
                        if hasattr(model.cache, 'memory_cache'):
                            cache_size = len(model.cache.memory_cache)
                            model.cache.memory_cache.clear()
                            cleared_items["embedding_cache"] += cache_size
                            
        except Exception as e:
            logger.warning(f"Embedding cache clear error: {e}")
        
        # 4. Python garbage collection
        try:
            collected = gc.collect()
            cleared_items["python_gc"] = collected
        except Exception as e:
            logger.warning(f"Garbage collection error: {e}")
        
        clear_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "cleared_items": cleared_items,
            "clear_time_seconds": round(clear_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return {
            "status": "error",
            "message": f"Cache clear failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# 2. GET /memory/usage - Resource monitoring
@app.get("/memory/usage")
async def get_memory_usage():
    """Detaylı memory kullanım bilgileri"""
    try:
        # System memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Process-specific memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Redis memory (eğer local ise)
        redis_memory = {}
        try:
            redis_info = await redis_client.info('memory')
            redis_memory = {
                "used_memory_mb": round(redis_info.get('used_memory', 0) / (1024*1024), 2),
                "used_memory_rss_mb": round(redis_info.get('used_memory_rss', 0) / (1024*1024), 2),
                "used_memory_peak_mb": round(redis_info.get('used_memory_peak', 0) / (1024*1024), 2)
            }
        except:
            redis_memory = {"status": "unavailable"}
        
        # Python memory usage
        python_memory = {
            "rss_mb": round(process_memory.rss / (1024*1024), 2),
            "vms_mb": round(process_memory.vms / (1024*1024), 2),
            "percent": round(process.memory_percent(), 2),
            "gc_objects": len(gc.get_objects())
        }
        
        # RAPTOR-specific memory estimation
        raptor_memory = {}
        try:
            if RA and RA.tree:
                raptor_memory = {
                    "tree_nodes": len(RA.tree.all_nodes),
                    "tree_layers": RA.tree.num_layers,
                    "estimated_tree_mb": round(len(RA.tree.all_nodes) * 0.1, 2)  # Rough estimate
                }
        except:
            raptor_memory = {"status": "unavailable"}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
                "free_gb": round(memory.free / (1024**3), 2)
            },
            "swap_memory": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "usage_percent": swap.percent
            },
            "process_memory": python_memory,
            "redis_memory": redis_memory,
            "raptor_memory": raptor_memory,
            "memory_alerts": _get_memory_alerts(memory, process)
        }
        
    except Exception as e:
        logger.error(f"Memory usage check failed: {e}")
        return {
            "status": "error",
            "message": f"Memory check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _get_memory_alerts(memory, process):
    """Memory usage alerts"""
    alerts = []
    
    if memory.percent > 90:
        alerts.append("CRITICAL: System memory usage > 90%")
    elif memory.percent > 80:
        alerts.append("WARNING: System memory usage > 80%")
    
    if process.memory_percent() > 50:
        alerts.append("WARNING: Process memory usage > 50%")
    
    available_gb = memory.available / (1024**3)
    if available_gb < 1:
        alerts.append("CRITICAL: Less than 1GB memory available")
    elif available_gb < 2:
        alerts.append("WARNING: Less than 2GB memory available")
    
    return alerts

# 3. DELETE /chat_history/{session_id} - Privacy compliance
@app.delete("/chat_history/{session_id}")
async def delete_chat_history(session_id: str):
    """Specific session'ın chat history'sini sil"""
    try:
        start_time = time.time()
        
        # Session ID validation
        if not session_id or len(session_id) < 5:
            return {
                "status": "error",
                "message": "Invalid session ID",
                "session_id": session_id
            }
        
        # Chat history key
        history_key = f"chat_history:{session_id}"
        
        # Check if session exists
        exists = await redis_client.exists(history_key)
        if not exists:
            return {
                "status": "warning",
                "message": "Session not found",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get message count before deletion
        message_count = await redis_client.llen(history_key)
        
        # Delete the session
        deleted = await redis_client.delete(history_key)
        
        # Also clean any related temporary data
        temp_patterns = [
            f"session_temp:{session_id}",
            f"cache:{session_id}:*",
            f"temp:{session_id}:*"
        ]
        
        additional_deleted = 0
        for pattern in temp_patterns:
            if ':*' in pattern:
                keys = await redis_client.keys(pattern)
                if keys:
                    additional_deleted += await redis_client.delete(*keys)
            else:
                additional_deleted += await redis_client.delete(pattern)
        
        delete_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "Chat history deleted successfully",
            "session_id": session_id,
            "deleted_messages": int(message_count),
            "additional_keys_deleted": additional_deleted,
            "delete_time_seconds": round(delete_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat history deletion failed for {session_id}: {e}")
        return {
            "status": "error",
            "message": f"Deletion failed: {str(e)}",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

# 4. GET /metrics/live - Real-time system health
@app.get("/metrics/live")
async def get_live_metrics():
    """Real-time sistem metrikleri ve sağlık durumu"""
    try:
        current_time = time.time()
        
        # Update active connections
        system_metrics.update_connections(len(manager.active_connections))
        
        # System resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Redis stats
        redis_stats = {}
        try:
            redis_info = await redis_client.info()
            redis_stats = {
                "connected_clients": redis_info.get('connected_clients', 0),
                "used_memory_mb": round(redis_info.get('used_memory', 0) / (1024*1024), 2),
                "total_commands_processed": redis_info.get('total_commands_processed', 0),
                "keyspace_hits": redis_info.get('keyspace_hits', 0),
                "keyspace_misses": redis_info.get('keyspace_misses', 0)
            }
            
            # Cache hit ratio
            hits = redis_stats.get('keyspace_hits', 0)
            misses = redis_stats.get('keyspace_misses', 0)
            total = hits + misses
            redis_stats['cache_hit_ratio'] = round((hits / max(total, 1)) * 100, 2)
            
        except Exception as e:
            redis_stats = {"status": "error", "message": str(e)}
        
        # RAPTOR performance
        raptor_stats = {}
        try:
            if RA and hasattr(RA, 'get_performance_summary'):
                perf_summary = RA.get_performance_summary()
                raptor_stats = {
                    "tree_ready": bool(RA.tree),
                    "total_nodes": len(RA.tree.all_nodes) if RA.tree else 0,
                    "tree_layers": RA.tree.num_layers if RA.tree else 0,
                    "performance_summary": perf_summary
                }
        except Exception as e:
            raptor_stats = {"status": "error", "message": str(e)}
        
        # Active session count
        try:
            session_pattern = "chat_history:*"
            session_keys = await redis_client.keys(session_pattern)
            active_sessions = len(session_keys)
        except:
            active_sessions = 0
        
        # Calculate rates
        uptime = system_metrics.get_uptime_seconds()
        time_since_reset = current_time - system_metrics.last_reset
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 1),
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            
            "system_resources": {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "cpu_usage_percent": round(cpu_percent, 1),
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
            },
            
            "application_metrics": {
                "active_websocket_connections": system_metrics.websocket_connections,
                "total_requests": system_metrics.request_count,
                "total_errors": system_metrics.error_count,
                "error_rate_percent": round((system_metrics.error_count / max(system_metrics.request_count, 1)) * 100, 2),
                "requests_per_second": round(system_metrics.request_count / max(time_since_reset, 1), 2),
                "raptor_queries": system_metrics.raptor_queries
            },
            
            "redis_metrics": redis_stats,
            "raptor_metrics": raptor_stats,
            
            "session_stats": {
                "active_sessions": active_sessions,
                "chat_sessions_created": system_metrics.chat_sessions
            },
            
            "health_status": _calculate_health_status(memory.percent, cpu_percent, redis_stats, raptor_stats)
        }
        
    except Exception as e:
        logger.error(f"Live metrics failed: {e}")
        return {
            "status": "error",
            "message": f"Metrics collection failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _calculate_health_status(memory_percent, cpu_percent, redis_stats, raptor_stats):
    """Overall system health calculation"""
    issues = []
    health_score = 100
    
    # Memory check
    if memory_percent > 90:
        issues.append("Critical memory usage")
        health_score -= 30
    elif memory_percent > 80:
        issues.append("High memory usage")
        health_score -= 15
    
    # CPU check
    if cpu_percent > 90:
        issues.append("Critical CPU usage")
        health_score -= 25
    elif cpu_percent > 80:
        issues.append("High CPU usage")
        health_score -= 10
    
    # Redis check
    if "error" in redis_stats.get("status", ""):
        issues.append("Redis connectivity issues")
        health_score -= 40
    
    # RAPTOR check
    if not raptor_stats.get("tree_ready", False):
        issues.append("RAPTOR tree not ready")
        health_score -= 35
    
    # Determine status
    if health_score >= 90:
        status = "excellent"
    elif health_score >= 75:
        status = "good"
    elif health_score >= 50:
        status = "degraded"
    else:
        status = "critical"
    
    return {
        "status": status,
        "health_score": max(health_score, 0),
        "issues": issues
    }

# 5. POST /raptor/optimize - Performance tuning
@app.post("/raptor/optimize")
async def optimize_raptor_performance():
    """RAPTOR sistemini optimize et"""
    try:
        start_time = time.time()
        optimization_results = {
            "cache_cleanup": False,
            "expired_cache_cleanup": False,
            "garbage_collection": False,
            "retriever_optimization": False,
            "embedding_cache_optimization": False
        }
        
        if not RA:
            return {
                "status": "error",
                "message": "RAPTOR system not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # 1. RAPTOR cache cleanup
        try:
            if hasattr(RA, 'clear_all_caches'):
                RA.clear_all_caches()
                optimization_results["cache_cleanup"] = True
        except Exception as e:
            logger.warning(f"RAPTOR cache cleanup error: {e}")
        
        # 2. Expired cache cleanup
        try:
            if hasattr(RA, 'retriever') and hasattr(RA.retriever, 'cleanup_expired_cache'):
                RA.retriever.cleanup_expired_cache()
                optimization_results["expired_cache_cleanup"] = True
        except Exception as e:
            logger.warning(f"Expired cache cleanup error: {e}")
        
        # 3. Retriever optimization
        try:
            if hasattr(RA, 'optimize_performance'):
                RA.optimize_performance()
                optimization_results["retriever_optimization"] = True
        except Exception as e:
            logger.warning(f"Retriever optimization error: {e}")
        
        # 4. Embedding model cache optimization
        try:
            if hasattr(RA, 'tree_builder') and hasattr(RA.tree_builder, 'embedding_models'):
                for model_name, model in RA.tree_builder.embedding_models.items():
                    if hasattr(model, 'cache') and model.cache:
                        # Clean expired entries
                        if hasattr(model.cache, 'cleanup_expired'):
                            model.cache.cleanup_expired()
                        # Optimize memory cache size
                        elif hasattr(model.cache, 'memory_cache'):
                            cache_size = len(model.cache.memory_cache)
                            if cache_size > 1000:  # If too large, keep only recent 500
                                # This is a simple optimization - keep most recent items
                                cache_items = list(model.cache.memory_cache.items())
                                model.cache.memory_cache.clear()
                                model.cache.memory_cache.update(dict(cache_items[-500:]))
                
                optimization_results["embedding_cache_optimization"] = True
                
        except Exception as e:
            logger.warning(f"Embedding cache optimization error: {e}")
        
        # 5. Python garbage collection
        try:
            collected = gc.collect()
            optimization_results["garbage_collection"] = collected
        except Exception as e:
            logger.warning(f"Garbage collection error: {e}")
        
        # Get performance stats after optimization
        performance_stats = {}
        try:
            if hasattr(RA, 'get_performance_summary'):
                performance_stats = RA.get_performance_summary()
        except:
            performance_stats = {"status": "unavailable"}
        
        optimization_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "RAPTOR optimization completed",
            "optimization_results": optimization_results,
            "optimization_time_seconds": round(optimization_time, 3),
            "performance_stats": performance_stats,
            "recommendations": _get_optimization_recommendations(performance_stats),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAPTOR optimization failed: {e}")
        return {
            "status": "error",
            "message": f"Optimization failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _get_optimization_recommendations(performance_stats):
    """RAPTOR performance based recommendations"""
    recommendations = []
    
    try:
        # Check cache hit rate
        if 'retriever' in performance_stats:
            cache_hit_rate = performance_stats['retriever'].get('cache_hit_rate', 0)
            if cache_hit_rate < 0.5:
                recommendations.append("Low cache hit rate - consider increasing cache TTL")
            elif cache_hit_rate > 0.9:
                recommendations.append("Excellent cache performance")
        
        # Check average query time
        if 'pipeline' in performance_stats:
            avg_query_time = performance_stats['pipeline'].get('avg_query_time', 0)
            if avg_query_time > 5:
                recommendations.append("High query times - consider enabling early termination")
            elif avg_query_time < 1:
                recommendations.append("Excellent query performance")
        
        # Check tree stats
        if 'tree_stats' in performance_stats:
            total_nodes = performance_stats['tree_stats'].get('total_nodes', 0)
            if total_nodes > 10000:
                recommendations.append("Large tree detected - monitor memory usage")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
            
    except Exception as e:
        recommendations.append(f"Unable to generate recommendations: {str(e)}")
    
    return recommendations

# ====================================================================
# ORIGINAL ENDPOINTS (Enhanced)
# ====================================================================

# Enhanced health check endpoint
@app.get("/health")
async def enhanced_health_check():
    """Enhanced sistem sağlık kontrolü with model loading status"""
    start_time = time.time()
    
    try:
        # Redis bağlantısını test et
        await redis_client.ping()
        redis_status = "healthy"
        redis_latency = (time.time() - start_time) * 1000
    except Exception as e:
        redis_status = "unhealthy"
        redis_latency = None
    
    # RAPTOR durumunu kontrol et
    raptor_status = "healthy" if RA and RA.tree else "unhealthy"
    
    # Model loading durumunu kontrol et
    model_status = await check_model_loading_status()
    
    # WebSocket connection sayısı
    active_connections = len(manager.active_connections)
    
    # System resources (optional)
    system_info = {}
    try:
        memory = psutil.virtual_memory()
        system_info = {
            'memory_usage_percent': round(memory.percent, 1),
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'cpu_usage_percent': round(psutil.cpu_percent(interval=0.1), 1)
        }
    except ImportError:
        pass
    
    # Overall status
    overall_healthy = (
        redis_status == "healthy" and 
        raptor_status == "healthy" and 
        model_status['models_loaded']
    )
    
    response = {
        "status": "healthy" if overall_healthy else "degraded" if model_status['models_loaded'] else "unhealthy",
        "timestamp": time.time(),
        "services": {
            "redis": redis_status,
            "raptor": raptor_status,
        },
        "models": {
            "models_loaded": model_status['models_loaded'],
            "loaded_models": model_status['loaded_models'],
            "loading_time_seconds": round(time.time() - model_status['loading_start_time'], 1),
            "loading_errors": model_status['loading_errors'][-3:] if model_status['loading_errors'] else []  # Son 3 hata
        },
        "performance": {
            "active_connections": active_connections,
            "redis_latency_ms": round(redis_latency, 2) if redis_latency else None,
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
    }
    
    # Optional system info ekleme
    if system_info:
        response["system"] = system_info
    
    return response

# Model loading monitoring endpoint
@app.get("/models/status")
async def model_loading_status():
    """Sadece model yükleme durumunu döndürür"""
    return await check_model_loading_status()

# Chat geçmişini getiren endpoint
@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    """Belirli bir session'ın chat geçmişini döndürür"""
    history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
    return [json.loads(msg) for msg in history]

# ====================================================================
# ENHANCED RETRIEVE FUNCTION (with metrics tracking)
# ====================================================================

async def batch_retrieve_async(queries: List[str]) -> Dict[str, any]:
    """
    Paralel retrieve + unique content filtering with metrics tracking
    """
    if not queries:
        return {"unique_contexts": [], "query_mapping": {}, "stats": {}}
    
    # Track RAPTOR query
    system_metrics.increment_raptor_query()
    
    # Her query için async task oluştur
    async def retrieve_single(query: str) -> tuple:
        try:
            if hasattr(RA, '_retrieve_async'):
                result = await RA._retrieve_async(query)
            else:
                # Fallback to sync version in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, RA.retrieve, query)
            return query, result
        except Exception as e:
            return query, f"Error retrieving '{query}': {str(e)}"
    
    # 🚀 PARALEL EXECUTION
    tasks = [retrieve_single(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 🔍 UNIQUE CONTENT FILTERING
    seen_contents = set()  # Raw text'leri track etmek için
    unique_contexts = []   # Unique content'ler
    query_mapping = {}     # Hangi context hangi query'den geldi
    duplicate_count = 0
    
    for query, result in results:
        if isinstance(result, Exception):
            continue
            
        # Raw text'i normalize et (boşlukları temizle)
        normalized_content = ' '.join(str(result).split())
        
        # 🎯 DUPLICATE CHECK
        if normalized_content not in seen_contents:
            seen_contents.add(normalized_content)
            unique_contexts.append(str(result))
            
            # İlk kez görülen content için mapping yap
            query_mapping[len(unique_contexts) - 1] = {
                "primary_query": query,
                "content_preview": normalized_content[:100] + "..." if len(normalized_content) > 100 else normalized_content
            }
        else:
            duplicate_count += 1
            # Duplicate bulundu, hangi query'den geldiğini mapping'e ekle
            for idx, mapping in query_mapping.items():
                # Bu content'in daha önce kaydedilen versiyonunu bul
                if unique_contexts[idx] and ' '.join(unique_contexts[idx].split()) == normalized_content:
                    if "additional_queries" not in mapping:
                        mapping["additional_queries"] = []
                    mapping["additional_queries"].append(query)
                    break
    
    return {
        "unique_contexts": unique_contexts,
        "query_mapping": query_mapping, 
        "stats": {
            "total_queries": len(queries),
            "total_results": len(results),
            "unique_contents": len(unique_contexts),
            "duplicates_removed": duplicate_count,
            "deduplication_ratio": f"{duplicate_count}/{len(results)}" if results else "0/0"
        }
    }

# ====================================================================
# STREAM RESPONSE PROCESSING (Enhanced with metrics)
# ====================================================================

async def process_stream_response(response_stream, session_id: str, client_id: str):
    """
    OpenAI stream response'unu işler ve tool calls varsa onları yönetir
    Enhanced with metrics tracking
    """
    chat_history = []
    full_content = ""
    collected_tool_calls = {}
    
    async for chunk in response_stream:
        delta = chunk.choices[0].delta
        
        # Tool calls kontrolü ve biriktirme
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index
                
                # İlk kez görüyorsak initialize et
                if index not in collected_tool_calls:
                    collected_tool_calls[index] = {
                        'id': '',
                        'type': 'function',
                        'function': {
                            'name': '',
                            'arguments': ''
                        }
                    }
                
                # ID varsa ekle
                if tool_call_delta.id:
                    collected_tool_calls[index]['id'] = tool_call_delta.id
                
                # Function bilgileri varsa ekle
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        collected_tool_calls[index]['function']['name'] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        collected_tool_calls[index]['function']['arguments'] += tool_call_delta.function.arguments
            
        # Content varsa ekle ve stream et
        if hasattr(delta, 'content') and delta.content:
            full_content += delta.content
            
            # Stream chunk'ını gönder
            await manager.send_stream_chunk({
                "type": "content_chunk",
                "content": delta.content
            }, client_id)
    
    # Eğer tool calls varsa işle
    if collected_tool_calls:
        # Tool calls tamamlandığını belirt
        await manager.send_stream_chunk({
            "type": "tool_calls_start",
            "message": "Dokümanlardan bilgi aranıyor..."
        }, client_id)
        
        # Collected tool calls'ları proper format'a çevir
        tool_calls_list = []
        for index, tool_call in collected_tool_calls.items():
            tool_calls_list.append({
                'id': tool_call['id'],
                'type': tool_call['type'],
                'function': tool_call['function']
            })
        
        # Tool calls'u chat history'ye ekle
        chat_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_list
        })
        
        # Her tool call'u işle
        for tool_call in tool_calls_list:
            function_name = tool_call['function']['name']
            
            try:
                function_args = json.loads(tool_call['function']['arguments'])
            except json.JSONDecodeError:
                logger.error(f"JSON decode error for arguments: {tool_call['function']['arguments']}")
                continue
            
            call_id = tool_call['id']
            
            # 🚀 DOKUMAN ARAMA TOOL'U
            if function_name == "get_document_info":
                queries = function_args.get('queries', [])
                
                if not queries:
                    # Fallback: tek query varsa
                    single_query = function_args.get('query', '')
                    if single_query:
                        queries = [single_query]
                
                if not queries:
                    chat_history.append({
                        "role": "tool",
                        "content": json.dumps({
                            "error": "Hiç arama sorgusu belirtilmedi"
                        }),
                        "tool_call_id": call_id
                    })
                    continue
                
                # Progress mesajı gönder
                query_text = ", ".join(queries[:3])  # İlk 3 query'yi göster
                if len(queries) > 3:
                    query_text += f" ve {len(queries) - 3} tane daha"
                
                await manager.send_stream_chunk({
                    "type": "rag_search",
                    "message": f"'{query_text}' hakkında dökümanlardan bilgi aranıyor..."
                }, client_id)
                
                try:
                    # 🚀 PARALEL RETRIEVE + UNIQUE FILTERING
                    start_time = asyncio.get_event_loop().time()
                    
                    retrieve_result = await batch_retrieve_async(queries)
                    unique_contexts = retrieve_result["unique_contexts"]
                    query_mapping = retrieve_result["query_mapping"]
                    stats = retrieve_result["stats"]
                    
                    end_time = asyncio.get_event_loop().time()
                    search_time = round((end_time - start_time) * 1000)  # milliseconds
                    
                    # 📊 DEDUPLICATION STATS
                    await manager.send_stream_chunk({
                        "type": "rag_complete",
                        "message": f"{stats['total_queries']} sorgu, {stats['unique_contents']} unique sonuç ({stats['duplicates_removed']} duplikat kaldırıldı) - {search_time}ms"
                    }, client_id)
                    
                    # 🔗 UNIQUE CONTENT BİRLEŞTİRME
                    combined_context = "\n\n--- DÖKÜMAN BİLGİLERİ ---\n\n"
                    
                    for i, context in enumerate(unique_contexts):
                        mapping = query_mapping.get(i, {})
                        primary_query = mapping.get("primary_query", f"Sorgu {i+1}")
                        additional_queries = mapping.get("additional_queries", [])
                        
                        # Hangi query'lerden geldiğini belirt
                        source_info = f"Birincil sorgu: {primary_query}"
                        if additional_queries:
                            source_info += f" (Ayrıca: {', '.join(additional_queries)})"
                        
                        combined_context += f"=== Bilgi Bölümü {i+1} ===\n"
                        combined_context += f"Kaynak: {source_info}\n"
                        combined_context += f"{context}\n\n"
                    
                    chat_history.append({
                        "role": "tool",
                        "content": json.dumps({
                            "queries": queries,
                            "unique_contexts": unique_contexts,
                            "combined_context": combined_context,
                            "query_mapping": query_mapping,
                            "stats": stats,
                            "search_time_ms": search_time
                        }),
                        "tool_call_id": call_id
                    })
                    
                except Exception as e:
                    logger.error(f"RAG search error: {e}")
                    chat_history.append({
                        "role": "tool", 
                        "content": json.dumps({
                            "queries": queries,
                            "error": f"Doküman arama sırasında hata oluştu: {str(e)}"
                        }),
                        "tool_call_id": call_id
                    })

        # Tool calls tamamlandıktan sonra final response'u stream et
        await manager.send_stream_chunk({
            "type": "final_response_start",
            "message": "Cevap hazırlanıyor..."
        }, client_id)
        
        # Final response için yeni stream
        try:
            final_response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=chat_history,
                temperature=0.7,
                stream=True
            )
            
            final_content = ""
            async for chunk in final_response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    final_content += delta.content
                    await manager.send_stream_chunk({
                        "type": "content_chunk",
                        "content": delta.content
                    }, client_id)
            
            # Final message'ı history'ye ekle
            chat_history.append({
                "role": "assistant",
                "content": final_content
            })
            
            # Redis'e kaydet
            await redis_client.rpush(f"chat_history:{session_id}", json.dumps(chat_history[-1]))
            
        except Exception as e:
            logger.error(f"Final response error: {e}")
            await manager.send_stream_chunk({
                "type": "content_chunk",
                "content": f"Cevap oluştururken bir hata oluştu: {str(e)}"
            }, client_id)
        
        # Stream tamamlandığını belirt
        await manager.send_stream_chunk({
            "type": "stream_end"
        }, client_id)
        
        return final_content if 'final_content' in locals() else ""
    
    else:
        # Tool calls yoksa direkt content'i kaydet
        if full_content:
            chat_history.append({
                "role": "assistant", 
                "content": full_content
            })
            
            # Redis'e kaydet
            await redis_client.rpush(f"chat_history:{session_id}", json.dumps(chat_history[-1]))
        
        # Stream tamamlandığını belirt
        await manager.send_stream_chunk({
            "type": "stream_end"
        }, client_id)
        
        return full_content

# ====================================================================
# MAIN WEBSOCKET ENDPOINT (Enhanced)
# ====================================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Ana WebSocket endpoint - Enhanced with metrics tracking"""
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        while True:
            user_message = await websocket.receive_text()
            
            # Kullanıcı mesajını Redis'e kaydet
            await redis_client.rpush(
                f"chat_history:{session_id}", 
                json.dumps({"role": "user", "content": user_message})
            )
            
            # Chat geçmişini al
            formatted_history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
            formatted_history = [json.loads(msg) for msg in formatted_history]            
            
            # Stream başladığını belirt
            await manager.send_stream_chunk({
                "type": "stream_start"
            }, client_id)
            
            try:
                # OpenAI API stream çağrısı
                response_stream = await openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": """
## ROL TANIMI
Sen akıllı bir soru-cevap asistanısın. Kullanıcıların sorularına yardımcı olmak için doküman veri tabanında arama yapabilir ve detaylı, doğru cevaplar verebilirsin.

## TEMEl GÖREVLERİN
1. **Genel Sorular**: Basit, günlük konuşma tarzı sorulara direkt cevap ver
2. **Spesifik Sorular**: Detaylı, teknik veya özel konularda bilgi gerektiren sorular için doküman araması yap
3. **Karmaşık Sorular**: Birden fazla konuyu kapsayan sorular için sistematik arama stratejisi kullan

## KİŞİLİK ÖZELLİKLERİN
- **Yardımsever ve samimi**: Kullanıcı dostu, anlaşılır dil kullan
- **Dikkatli**: Doğru bilgi vermeye odaklan
- **Verimli**: Gereksiz uzatmadan konuya odaklan
- **Şeffaf**: Bilgi kaynaklarını belirt, emin olmadığın durumlarda bunu söyle

## CEVAP VERME STRATEJİN

### Basit/Genel Sorularda:
- Direkt cevap ver (tool kullanma)
- Kısa ve net ol
- Günlük dil kullan

Örnek sorular:
- "Merhaba, nasılsın?"
- "Teşekkürler"
- "Bugün hava nasıl?"

### Spesifik/Teknik Sorularda:
1. **Tool Kullan**: Dokümanlardan bilgi ara
2. **Özetle**: Bulunan bilgiyi kendi deyimlerinle açıkla
3. **Kaynak Belirt**: Bilginin nereden geldiğini söyle

### Karmaşık Sorularda:
1. **Sistematik Yaklaşım**: Soruyu parçalara böl
2. **Çoklu Arama**: Her parça için spesifik arama yap
3. **Bütünleştir**: Sonuçları mantıklı bir şekilde birleştir

**Örnek Query Oluşturma:**
Kullanıcı: "X konusunda nasıl başlarım ve hangi kaynakları önerirsin?"

Tool'a gönderilecek queryler:
1. "X konusuna giriş rehberi"
2. "X öğrenme kaynakları"
3. "X başlangıç adımları"
4. "X en iyi uygulamalar"

## TOOL KULLANIM KURALLARI

### Ne Zaman Tool Kullanacaksın:
✅ Teknik sorular
✅ Spesifik bilgi isteyen sorular  
✅ "Nasıl", "Ne", "Hangi" ile başlayan detay gerektiren sorular
✅ Tarihsel bilgiler
✅ Prosedür/süreç soruları

### Ne Zaman Tool Kullanmayacaksın:
❌ Selamlaşma, teşekkür, kibarlık ifadeleri
❌ "Nasılsın?", "İyi akşamlar" gibi sohbet
❌ Genel nezaket konuşmaları

### Query Oluşturma İpuçları:
- **Spesifik ol**: "genel bilgi" yerine "X'in tanımı"
- **Anahtar kelimeler kullan**: Önemli terimleri dahil et
- **Kapsamlı sorular için parçala**: Her alt konu için ayrı query
- **Net ve kısa**: 2-8 kelime ideal

## İLETİŞİM TARZI
- Günlük konuşma dili kullan
- Kısa ve anlaşılır cümleler
- Jargondan kaçın, basit açıkla
- Kullanıcıyı düşünmeye teşvik et
- "Sen ne düşünüyorsun?" tarzı etkileşim

## ÖRNEK YAKLAŞIMLAR

### Basit Soru:
Kullanıcı: "Merhaba!"
Sen: "Merhaba! Size nasıl yardımcı olabilirim?"

### Spesifik Soru:
Kullanıcı: "Python'da liste nasıl oluşturulur?"
Sen: (Tool kullan) → Bulduğun bilgiyi özetle

### Karmaşık Soru:
Kullanıcı: "Web geliştirme nasıl öğrenirim?"
Sen: (Çoklu tool kullan: "web geliştirme başlangıç", "web programlama dilleri", "web geliştirme yol haritası")

## HATIRLATMALAR
- Her durumda yardımcı olmaya çalış
- Bilmediğin konularda dürüst ol
- Bulunan bilgiyi kendi sözlerinle açıkla
- Kaynakları direkt kopyalama, yorumla
- Kullanıcının seviyesine uygun açıkla

## KAÇINILACAKLAR
❌ Çok uzun akademik açıklamalar
❌ Gereksiz teknik detaylar
❌ Belirsiz cevaplar
❌ Kaynak bilgisini direkt yapıştırma
❌ "Bu konuda bilgim yok" demek (önce ara!)

Unutma: Sen bir bilgi köprüsüsün. Kullanıcı ile dokümanlar arasında akıllı bir bağlantı kuruyorsun.
                            """
                        }
                    ] + formatted_history,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "get_document_info",
                                "description": "Doküman veri tabanından bilgi aramak için kullanılır. Spesifik, teknik veya detaylı bilgi gerektiren sorular için kullan. Karmaşık sorular için birden fazla arama sorgusu oluşturabilirsin.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "queries": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            },
                                            "description": "Kullanıcının sorusuna cevap vermek için dokümanlardan arama yapılacak sorgu listesi. Kapsamlı konular için birden fazla spesifik sorgu oluştur. Her sorgu kısa ve odaklı olmalı.",
                                            "minItems": 1,
                                            "maxItems": 5
                                        }
                                    },
                                    "required": ["queries"]
                                }
                            }
                        },
                    ],
                    temperature=0.7,
                    stream=True,
                )
                
                # Stream response'u işle
                await process_stream_response(response_stream, session_id, client_id)
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                system_metrics.increment_error()
                await manager.send_stream_chunk({
                    "type": "content_chunk",
                    "content": f"Üzgünüm, bir hata oluştu: {str(e)}"
                }, client_id)
                await manager.send_stream_chunk({
                    "type": "stream_end"
                }, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        system_metrics.increment_error()
        manager.disconnect(client_id)

# ====================================================================
# STARTUP & SHUTDOWN EVENTS
# ====================================================================

@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında model loading tracking'i başlat"""
    global MODEL_LOADING_STATE
    MODEL_LOADING_STATE['loading_start_time'] = time.time()
    
    logger.info("🚀 Enhanced RAPTOR QA Server starting up...")
    logger.info("📊 Management endpoints enabled")
    
    # Background task olarak model status check'i çalıştır
    asyncio.create_task(periodic_model_check())

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    logger.info("🛑 Server shutting down...")
    
    # Close Redis connections
    try:
        await redis_client.close()
        logger.info("✅ Redis connections closed")
    except Exception as e:
        logger.error(f"Redis shutdown error: {e}")

async def periodic_model_check():
    """Periyodik olarak model durumunu kontrol et"""
    while True:
        try:
            await check_model_loading_status()
            await asyncio.sleep(30)  # Her 30 saniyede bir kontrol et
        except Exception as e:
            logger.error(f"Periodic model check error: {e}")
            await asyncio.sleep(60)  # Hata durumunda 60 saniye bekle

# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )