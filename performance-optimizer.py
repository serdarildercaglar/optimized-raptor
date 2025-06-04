"""
DOSYA: performance-optimizer.py
AÇIKLAMA: Enhanced RAPTOR-specific performance testing and optimization
"""

import os
import sys
import time
import asyncio
import statistics
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import logging
import threading
import queue
import requests
from pathlib import Path
import traceback

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not installed. Run: pip install psutil")

# WebSocket support
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️ websockets not installed. Run: pip install websockets")

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Enhanced performance test metrics for RAPTOR"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    # RAPTOR-specific metrics
    stream_responses: int = 0
    tool_call_responses: int = 0
    rag_search_count: int = 0
    complete_conversations: int = 0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / max(self.total_requests, 1)) * 100
    
    @property
    def requests_per_second(self) -> float:
        return self.total_requests / max(self.duration_seconds, 0.1)
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def median_response_time(self) -> float:
        return statistics.median(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'timeout_requests': self.timeout_requests,
            'success_rate_percent': round(self.success_rate, 2),
            'duration_seconds': round(self.duration_seconds, 2),
            'requests_per_second': round(self.requests_per_second, 2),
            'raptor_metrics': {
                'stream_responses': self.stream_responses,
                'tool_call_responses': self.tool_call_responses,
                'rag_search_count': self.rag_search_count,
                'complete_conversations': self.complete_conversations
            },
            'response_times': {
                'avg_ms': round(self.avg_response_time * 1000, 2),
                'median_ms': round(self.median_response_time * 1000, 2),
                'p95_ms': round(self.p95_response_time * 1000, 2),
                'p99_ms': round(self.p99_response_time * 1000, 2),
                'min_ms': round(min(self.response_times) * 1000, 2) if self.response_times else 0,
                'max_ms': round(max(self.response_times) * 1000, 2) if self.response_times else 0
            },
            'resource_usage': {
                'peak_memory_mb': round(self.peak_memory_mb, 2),
                'peak_cpu_percent': round(self.peak_cpu_percent, 2),
                'avg_memory_mb': round(self.avg_memory_mb, 2),
                'avg_cpu_percent': round(self.avg_cpu_percent, 2)
            },
            'error_count': len(self.error_messages),
            'unique_errors': len(set(self.error_messages))
        }

class ResourceMonitor:
    """Real-time resource usage monitoring during tests"""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = None
    
    def start(self):
        """Start resource monitoring"""
        if not PSUTIL_AVAILABLE:
            logger.warning("Resource monitoring disabled (psutil not available)")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self) -> Tuple[float, float, float, float]:
        """Stop monitoring and return peak/avg values"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        if not self.memory_samples or not self.cpu_samples:
            return 0, 0, 0, 0
        
        return (
            max(self.memory_samples),
            max(self.cpu_samples),
            statistics.mean(self.memory_samples),
            statistics.mean(self.cpu_samples)
        )
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
            
            time.sleep(2)

class RAPTORWebSocketTester:
    """RAPTOR-specific WebSocket load testing"""
    
    def __init__(self, url: str):
        self.url = url
        self.metrics = PerformanceMetrics()
        self.resource_monitor = ResourceMonitor()
    
    async def run_websocket_test(self, 
                                concurrent_connections: int = 10,
                                messages_per_connection: int = 5,
                                test_duration_seconds: int = 120) -> PerformanceMetrics:
        """Run RAPTOR WebSocket load test"""
        
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSocket testing requires 'websockets' package")
            return self.metrics
        
        logger.info(f"🔄 Starting RAPTOR WebSocket load test...")
        logger.info(f"   Connections: {concurrent_connections}")
        logger.info(f"   Messages per connection: {messages_per_connection}")
        logger.info(f"   Duration: {test_duration_seconds}s")
        
        self.metrics.start_time = time.time()
        self.resource_monitor.start()
        
        # RAPTOR-specific test messages
        test_messages = [
            "Bu doküman hakkında kısa bir özet çıkarabilir misin?",
            "Dokümanın ana konularını listeler misin?",
            "En önemli bilgiler nelerdir?",
            "Bu konuda detaylı bilgi verebilir misin?",
            "Hangi konularda daha fazla açıklama yapabilirsin?",
            "Bu dokümanın öne çıkan noktaları neler?",
            "Konuyla ilgili örnekler var mı?",
            "Bu bilgileri nasıl uygulayabilirim?",
            "İlgili diğer konular nelerdir?",
            "Özet olarak ne söyleyebilirsin?"
        ]
        
        # Create connection tasks
        tasks = []
        for i in range(concurrent_connections):
            task = self._websocket_connection_worker(
                f"client_{i}",
                test_messages[:messages_per_connection],
                test_duration_seconds
            )
            tasks.append(task)
        
        # Run all connections concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.metrics.end_time = time.time()
        
        # Stop resource monitoring
        peak_mem, peak_cpu, avg_mem, avg_cpu = self.resource_monitor.stop()
        self.metrics.peak_memory_mb = peak_mem
        self.metrics.peak_cpu_percent = peak_cpu
        self.metrics.avg_memory_mb = avg_mem
        self.metrics.avg_cpu_percent = avg_cpu
        
        return self.metrics
    
    async def _websocket_connection_worker(self, 
                                         client_id: str, 
                                         messages: List[str],
                                         max_duration: int):
        """Single WebSocket connection worker for RAPTOR testing"""
        
        start_time = time.time()
        
        try:
            async with websockets.connect(self.url) as websocket:
                message_index = 0
                
                while (time.time() - start_time) < max_duration:
                    # Send message
                    message = messages[message_index % len(messages)]
                    message_start = time.time()
                    
                    try:
                        await websocket.send(message)
                        
                        # RAPTOR-specific response handling
                        response_received = False
                        stream_started = False
                        tool_calls_detected = False
                        rag_searches = 0
                        message_parts = []
                        
                        # Wait for complete conversation
                        while not response_received and (time.time() - message_start) < 60:  # 60s timeout
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                                response_data = json.loads(response)
                                message_type = response_data.get('type', '')
                                
                                # Handle different RAPTOR message types
                                if message_type == 'stream_start':
                                    stream_started = True
                                    
                                elif message_type == 'content_chunk':
                                    content = response_data.get('content', '')
                                    if content:
                                        message_parts.append(content)
                                
                                elif message_type == 'tool_calls_start':
                                    tool_calls_detected = True
                                
                                elif message_type == 'rag_search':
                                    rag_searches += 1
                                
                                elif message_type == 'rag_complete':
                                    pass  # RAG search completed
                                
                                elif message_type == 'final_response_start':
                                    pass  # Final response starting
                                
                                elif message_type == 'stream_end':
                                    response_received = True
                                    response_time = time.time() - message_start
                                    
                                    # Record successful conversation
                                    self.metrics.total_requests += 1
                                    self.metrics.successful_requests += 1
                                    self.metrics.response_times.append(response_time)
                                    
                                    # Record RAPTOR-specific metrics
                                    if stream_started:
                                        self.metrics.stream_responses += 1
                                    if tool_calls_detected:
                                        self.metrics.tool_call_responses += 1
                                    if rag_searches > 0:
                                        self.metrics.rag_search_count += rag_searches
                                    if len(message_parts) > 0:
                                        self.metrics.complete_conversations += 1
                                
                            except asyncio.TimeoutError:
                                continue
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error: {e}")
                                continue
                            except Exception as e:
                                logger.warning(f"Message handling error: {e}")
                                continue
                        
                        # Check if conversation timed out
                        if not response_received:
                            self.metrics.total_requests += 1
                            self.metrics.timeout_requests += 1
                            self.metrics.error_messages.append("Conversation timeout")
                        
                    except Exception as e:
                        self.metrics.total_requests += 1
                        self.metrics.failed_requests += 1
                        self.metrics.error_messages.append(str(e))
                    
                    message_index += 1
                    
                    # Brief delay between messages
                    await asyncio.sleep(2)  # 2 seconds between messages
        
        except Exception as e:
            logger.error(f"WebSocket connection failed for {client_id}: {e}")
            self.metrics.error_messages.append(f"Connection failed: {str(e)}")

class RAPTORHTTPTester:
    """RAPTOR-specific HTTP endpoint testing"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.metrics = PerformanceMetrics()
        self.resource_monitor = ResourceMonitor()
    
    def run_http_test(self,
                     concurrent_requests: int = 20,
                     total_requests: int = 200) -> PerformanceMetrics:
        """Run RAPTOR HTTP load test"""
        
        logger.info(f"🔄 Starting RAPTOR HTTP load test...")
        logger.info(f"   Concurrent requests: {concurrent_requests}")
        logger.info(f"   Total requests: {total_requests}")
        
        self.metrics.start_time = time.time()
        self.resource_monitor.start()
        
        # RAPTOR-specific endpoints
        endpoints = [
            '/health',
            '/models/status'
        ]
        
        # Create request tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            
            for i in range(total_requests):
                endpoint = endpoints[i % len(endpoints)]
                future = executor.submit(self._make_request, endpoint)
                futures.append(future)
            
            # Wait for all requests to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Request failed: {e}")
        
        self.metrics.end_time = time.time()
        
        # Stop resource monitoring
        peak_mem, peak_cpu, avg_mem, avg_cpu = self.resource_monitor.stop()
        self.metrics.peak_memory_mb = peak_mem
        self.metrics.peak_cpu_percent = peak_cpu
        self.metrics.avg_memory_mb = avg_mem
        self.metrics.avg_cpu_percent = avg_cpu
        
        return self.metrics
    
    def _make_request(self, endpoint: str):
        """Make single HTTP request with RAPTOR-specific handling"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=30)
            response_time = time.time() - start_time
            
            self.metrics.total_requests += 1
            self.metrics.response_times.append(response_time)
            
            if response.status_code == 200:
                self.metrics.successful_requests += 1
                
                # RAPTOR-specific response validation
                if endpoint == '/health':
                    try:
                        health_data = response.json()
                        status = health_data.get('status', 'unknown')
                        if status not in ['healthy', 'degraded']:
                            self.metrics.error_messages.append(f"Unhealthy status: {status}")
                    except:
                        self.metrics.error_messages.append("Invalid health response format")
                        
            else:
                self.metrics.failed_requests += 1
                self.metrics.error_messages.append(f"HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.metrics.total_requests += 1
            self.metrics.timeout_requests += 1
            self.metrics.error_messages.append("Request timeout")
            
        except Exception as e:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.error_messages.append(str(e))

class RAPTORPerformanceAnalyzer:
    """RAPTOR-specific performance analysis and recommendations"""
    
    @staticmethod
    def analyze_metrics(metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze RAPTOR performance metrics and provide recommendations"""
        
        analysis = {
            'performance_grade': 'A',
            'bottlenecks': [],
            'recommendations': [],
            'warnings': [],
            'optimizations': []
        }
        
        # RAPTOR-specific response time analysis (AI systems are slower)
        avg_response_ms = metrics.avg_response_time * 1000
        p95_response_ms = metrics.p95_response_time * 1000
        
        if avg_response_ms > 15000:  # 15 seconds for AI
            analysis['performance_grade'] = 'F'
            analysis['bottlenecks'].append('Very slow AI response times')
            analysis['recommendations'].append('Check RAPTOR tree size and embedding model performance')
        elif avg_response_ms > 10000:  # 10 seconds
            analysis['performance_grade'] = 'D'
            analysis['bottlenecks'].append('Slow AI response times')
            analysis['recommendations'].append('Enable early termination and optimize batch size')
        elif avg_response_ms > 7000:  # 7 seconds
            analysis['performance_grade'] = 'C'
            analysis['recommendations'].append('Consider increasing concurrent operations')
        elif avg_response_ms > 5000:  # 5 seconds
            analysis['performance_grade'] = 'B'
        
        # RAPTOR-specific success rate analysis
        if metrics.success_rate < 80:  # Lower threshold for AI systems
            analysis['performance_grade'] = 'F'
            analysis['bottlenecks'].append('High error rate in AI conversations')
            analysis['recommendations'].append('Check RAPTOR tree integrity and Redis connectivity')
        elif metrics.success_rate < 90:
            analysis['warnings'].append('Moderate AI conversation failure rate')
        
        # Throughput analysis (lower expectations for AI)
        if metrics.requests_per_second < 0.5:  # Very low for AI
            analysis['bottlenecks'].append('Very low AI throughput')
            analysis['recommendations'].append('Scale up hardware or optimize RAPTOR configuration')
        elif metrics.requests_per_second < 2:
            analysis['recommendations'].append('Consider async processing improvements')
        
        # RAPTOR-specific metrics analysis
        if metrics.tool_call_responses > 0:
            tool_call_ratio = metrics.tool_call_responses / max(metrics.successful_requests, 1)
            if tool_call_ratio > 0.8:
                analysis['optimizations'].append('High RAG usage - consider cache optimization')
        
        if metrics.timeout_requests > metrics.successful_requests * 0.1:
            analysis['bottlenecks'].append('High timeout rate in AI conversations')
            analysis['recommendations'].append('Increase WebSocket timeout and optimize retrieval')
        
        # Resource usage analysis
        if metrics.peak_memory_mb > 8000:  # 8GB
            analysis['warnings'].append('High memory usage')
            analysis['optimizations'].append('Enable memory-optimized configuration')
        
        if metrics.peak_cpu_percent > 90:
            analysis['warnings'].append('High CPU usage')
            analysis['optimizations'].append('Increase worker processes or optimize batch sizes')
        
        # P95 vs average analysis for AI systems
        if p95_response_ms > avg_response_ms * 4:  # Higher variance acceptable for AI
            analysis['bottlenecks'].append('High response time variance in AI')
            analysis['recommendations'].append('Enable request timeout and improve caching')
        
        return analysis
    
    @staticmethod
    def generate_optimization_config(analysis: Dict[str, Any], 
                                   base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized RAPTOR configuration based on analysis"""
        
        optimized = base_config.copy()
        
        if 'Very slow AI response times' in analysis['bottlenecks']:
            # Aggressive optimizations for slow RAPTOR performance
            optimized.update({
                'tb_batch_size': min(base_config.get('tb_batch_size', 100) * 2, 300),
                'max_concurrent_operations': min(base_config.get('max_concurrent_operations', 10) + 6, 20),
                'tr_early_termination': True,
                'tr_confidence_threshold': 0.6,  # Lower for speed
                'cache_ttl': 7200,
                'tr_top_k': 5  # Reduce retrieval scope
            })
        
        elif 'Slow AI response times' in analysis['bottlenecks']:
            # Moderate optimizations
            optimized.update({
                'tb_batch_size': int(base_config.get('tb_batch_size', 100) * 1.5),
                'max_concurrent_operations': base_config.get('max_concurrent_operations', 10) + 3,
                'tr_early_termination': True,
                'cache_ttl': 3600
            })
        
        if 'High memory usage' in analysis['warnings']:
            # Memory optimizations for RAPTOR
            optimized.update({
                'tb_batch_size': max(base_config.get('tb_batch_size', 100) // 2, 25),
                'max_concurrent_operations': max(base_config.get('max_concurrent_operations', 10) // 2, 4),
                'cache_ttl': 1800,  # 30 minutes
                'tr_top_k': 3  # Reduce memory usage
            })
        
        if 'Very low AI throughput' in analysis['bottlenecks']:
            # Throughput optimizations for RAPTOR
            optimized.update({
                'workers': min(base_config.get('workers', 4) * 2, 8),
                'tb_batch_size': min(base_config.get('tb_batch_size', 100) * 2, 250),
                'max_concurrent_operations': min(base_config.get('max_concurrent_operations', 10) * 2, 16),
                'enable_async': True
            })
        
        # RAPTOR-specific optimizations
        if 'High RAG usage' in [opt for opt in analysis.get('optimizations', [])]:
            optimized.update({
                'tr_enable_caching': True,
                'cache_ttl': 7200,  # 2 hours for RAG cache
                'tr_adaptive_retrieval': True
            })
        
        return optimized

def run_comprehensive_raptor_test(server_url: str = "http://localhost:8000",
                                 websocket_url: str = "ws://localhost:8000/ws/test_client",
                                 output_file: str = None) -> Dict[str, Any]:
    """Run comprehensive RAPTOR load test suite"""
    
    results = {
        'timestamp': time.time(),
        'server_url': server_url,
        'websocket_url': websocket_url,
        'raptor_version': 'optimized',
        'tests': {}
    }
    
    logger.info("🚀 Starting comprehensive RAPTOR load test suite...")
    
    # Test 1: HTTP Load Test
    logger.info("\n📊 Test 1: RAPTOR HTTP Load Test")
    http_tester = RAPTORHTTPTester(server_url)
    http_metrics = http_tester.run_http_test(concurrent_requests=15, total_requests=150)
    results['tests']['http'] = http_metrics.to_dict()
    
    logger.info(f"   Success Rate: {http_metrics.success_rate:.1f}%")
    logger.info(f"   Avg Response: {http_metrics.avg_response_time*1000:.1f}ms")
    logger.info(f"   Throughput: {http_metrics.requests_per_second:.1f} req/s")
    
    # Test 2: WebSocket Load Test (RAPTOR-specific)
    logger.info("\n🔗 Test 2: RAPTOR WebSocket AI Conversation Test")
    try:
        if WEBSOCKETS_AVAILABLE:
            ws_tester = RAPTORWebSocketTester(websocket_url)
            ws_metrics = asyncio.run(ws_tester.run_websocket_test(
                concurrent_connections=8,
                messages_per_connection=4,
                test_duration_seconds=120
            ))
            results['tests']['websocket'] = ws_metrics.to_dict()
            
            logger.info(f"   Success Rate: {ws_metrics.success_rate:.1f}%")
            logger.info(f"   Avg Response: {ws_metrics.avg_response_time*1000:.1f}ms")
            logger.info(f"   Total Conversations: {ws_metrics.total_requests}")
            logger.info(f"   RAG Searches: {ws_metrics.rag_search_count}")
            logger.info(f"   Complete Conversations: {ws_metrics.complete_conversations}")
        else:
            results['tests']['websocket'] = {'error': 'websockets package not available'}
            
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        results['tests']['websocket'] = {'error': str(e)}
    
    # RAPTOR Performance Analysis
    logger.info("\n📈 RAPTOR Performance Analysis")
    if 'websocket' in results['tests'] and 'error' not in results['tests']['websocket']:
        # Recreate metrics object for analysis
        ws_metrics_dict = results['tests']['websocket']
        analysis_metrics = PerformanceMetrics()
        analysis_metrics.total_requests = ws_metrics_dict['total_requests']
        analysis_metrics.successful_requests = ws_metrics_dict['successful_requests']
        analysis_metrics.failed_requests = ws_metrics_dict['failed_requests']
        analysis_metrics.timeout_requests = ws_metrics_dict['timeout_requests']
        analysis_metrics.response_times = [t/1000 for t in [
            ws_metrics_dict['response_times']['avg_ms'],
            ws_metrics_dict['response_times']['median_ms']
        ] if t > 0]
        analysis_metrics.tool_call_responses = ws_metrics_dict['raptor_metrics']['tool_call_responses']
        analysis_metrics.peak_memory_mb = ws_metrics_dict['resource_usage']['peak_memory_mb']
        analysis_metrics.peak_cpu_percent = ws_metrics_dict['resource_usage']['peak_cpu_percent']
        
        analysis = RAPTORPerformanceAnalyzer.analyze_metrics(analysis_metrics)
        results['analysis'] = analysis
        
        logger.info(f"   Performance Grade: {analysis['performance_grade']}")
        if analysis['bottlenecks']:
            logger.warning(f"   Bottlenecks: {', '.join(analysis['bottlenecks'])}")
        if analysis['recommendations']:
            logger.info(f"   Top Recommendation: {analysis['recommendations'][0]}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Results saved to: {output_file}")
    
    return results

def optimize_raptor_config(test_results: Dict[str, Any],
                          current_config_path: str = "config/production.json",
                          output_config_path: str = "config/production_optimized.json"):
    """Generate optimized RAPTOR configuration based on test results"""
    
    logger.info("🔧 Generating optimized RAPTOR configuration...")
    
    # Load current config
    base_config = {}
    if os.path.exists(current_config_path):
        with open(current_config_path, 'r') as f:
            base_config = json.load(f)
    
    # Default RAPTOR config if none exists
    if not base_config:
        base_config = {
            'tb_batch_size': 100,
            'max_concurrent_operations': 10,
            'workers': 4,
            'cache_ttl': 3600,
            'tr_early_termination': True,
            'tr_top_k': 8,
            'tr_enable_caching': True,
            'enable_async': True
        }
    
    # Generate optimizations
    if 'analysis' in test_results:
        optimized_config = RAPTORPerformanceAnalyzer.generate_optimization_config(
            test_results['analysis'],
            base_config
        )
        
        # Save optimized config
        os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
        with open(output_config_path, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        
        logger.info(f"✅ Optimized RAPTOR config saved to: {output_config_path}")
        
        # Show changes
        changes = []
        for key, value in optimized_config.items():
            if key in base_config and base_config[key] != value:
                changes.append(f"{key}: {base_config[key]} → {value}")
        
        if changes:
            logger.info("📝 RAPTOR configuration changes:")
            for change in changes:
                logger.info(f"   {change}")
        else:
            logger.info("ℹ️ No RAPTOR configuration changes recommended")
        
        return optimized_config
    
    else:
        logger.warning("No analysis results available for RAPTOR optimization")
        return base_config

def main():
    """Main RAPTOR performance testing entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAPTOR Performance Testing & Optimization")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="RAPTOR server URL for testing")
    parser.add_argument("--websocket-url", default="ws://localhost:8000/ws/test_client",
                       help="RAPTOR WebSocket URL for testing")
    parser.add_argument("--output", default="raptor_performance_results.json",
                       help="Output file for results")
    parser.add_argument("--optimize", action="store_true",
                       help="Generate optimized RAPTOR configuration")
    parser.add_argument("--test-only", action="store_true",
                       help="Run tests only, don't optimize")
    
    args = parser.parse_args()
    
    # Run RAPTOR load tests
    results = run_comprehensive_raptor_test(
        server_url=args.server_url,
        websocket_url=args.websocket_url,
        output_file=args.output
    )
    
    # Generate RAPTOR optimization report
    if not args.test_only and args.optimize:
        optimized_config = optimize_raptor_config(results)
    
    # Print RAPTOR summary
    print("\n" + "="*60)
    print("📊 RAPTOR PERFORMANCE TEST SUMMARY")
    print("="*60)
    
    if 'http' in results['tests']:
        http_data = results['tests']['http']
        print(f"HTTP Test:")
        print(f"  Success Rate: {http_data.get('success_rate_percent', 0):.1f}%")
        print(f"  Throughput: {http_data.get('requests_per_second', 0):.1f} req/s")
        print(f"  Avg Response: {http_data.get('response_times', {}).get('avg_ms', 0):.1f}ms")
    
    if 'websocket' in results['tests'] and 'error' not in results['tests']['websocket']:
        ws_data = results['tests']['websocket']
        print(f"RAPTOR AI Conversation Test:")
        print(f"  Success Rate: {ws_data.get('success_rate_percent', 0):.1f}%")
        print(f"  Avg Response: {ws_data.get('response_times', {}).get('avg_ms', 0):.1f}ms")
        print(f"  Total Conversations: {ws_data.get('total_requests', 0)}")
        print(f"  RAG Searches: {ws_data.get('raptor_metrics', {}).get('rag_search_count', 0)}")
        print(f"  Complete Conversations: {ws_data.get('raptor_metrics', {}).get('complete_conversations', 0)}")
    
    if 'analysis' in results:
        analysis = results['analysis']
        print(f"RAPTOR Performance Grade: {analysis.get('performance_grade', 'N/A')}")
        if analysis.get('recommendations'):
            print(f"Top Recommendation: {analysis['recommendations'][0]}")
    
    print("="*60)

if __name__ == "__main__":
    main()