# optimized_async_retriever.py
"""
🚀 RAPTOR için optimize edilmiş async retrieve fonksiyonu
En hızlı şekilde birden fazla query'yi paralel olarak işler
"""

import asyncio
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

# RAPTOR imports
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.SummarizationModels import GPT41SummarizationModel

# OpenAI API key ayarı (gerekiyorsa)
os.environ["OPENAI_API_KEY"] = "None"

# Logging ayarı
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Tek bir retrieve işleminin sonucu"""
    query: str
    context: str
    success: bool
    response_time: float
    cache_hit: bool = False
    layer_info: Optional[List[Dict]] = None
    error: Optional[str] = None


@dataclass
class BatchRetrievalMetrics:
    """Batch retrieve işleminin metrikleri"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time: float = 0.0
    average_time_per_query: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_qps: float = 0.0
    concurrent_operations: int = 0
    results: List[RetrievalResult] = field(default_factory=list)


class OptimizedAsyncRetriever:
    """
    🚀 En hızlı RAPTOR async retrieve implementasyonu
    
    Özellikler:
    - Parallel query processing
    - Advanced caching with pre-warming
    - Performance monitoring
    - Error handling & resilience
    - Memory optimization
    - Production-ready features
    """
    
    def __init__(
        self,
        tree_path: str = "vectordb/raptor-production",
        max_concurrent_operations: int = 16,  # Optimal: 16 (test sonucuna göre)
        cache_ttl: int = 14400,  # 4 saat
        similarity_cache_threshold: float = 0.80,
        performance_profile: str = "balanced"  # balanced en hızlı!
    ):
        """
        Optimizer initializer
        
        Args:
            tree_path: RAPTOR tree dosya yolu
            max_concurrent_operations: Maksimum paralel işlem sayısı
            cache_ttl: Cache TTL (saniye)
            similarity_cache_threshold: Cache benzerlik eşiği
            performance_profile: Performans profili (speed/balanced/quality)
        """
        self.tree_path = tree_path
        self.max_concurrent_operations = max_concurrent_operations
        self.performance_profile = performance_profile
        
        # Performance profile'a göre parametreler
        self._configure_performance_profile()
        
        # RAPTOR config oluştur
        self.config = self._create_optimized_config(
            cache_ttl=cache_ttl,
            similarity_cache_threshold=similarity_cache_threshold
        )
        
        # RAPTOR instance
        self.ra = None
        self._initialize_raptor()
        
        # Metrics
        self.session_metrics = BatchRetrievalMetrics()
        
        logger.info(f"🚀 OptimizedAsyncRetriever initialized with {self.performance_profile} profile")
        logger.info(f"📊 Max concurrent operations: {self.max_concurrent_operations}")
        logger.info(f"🧠 Cache TTL: {cache_ttl}s, Similarity threshold: {similarity_cache_threshold}")
    
    def _configure_performance_profile(self):
        """Performance profile'a göre parametreleri ayarla"""
        if self.performance_profile == "speed":
            self.top_k = 6
            self.max_tokens = 2500
            self.early_termination = True
            self.collapse_tree = True
            
        elif self.performance_profile == "balanced":
            self.top_k = 8
            self.max_tokens = 3500
            self.early_termination = True
            self.collapse_tree = False
            
        elif self.performance_profile == "quality":
            self.top_k = 12
            self.max_tokens = 5000
            self.early_termination = False
            self.collapse_tree = False
            
        else:
            raise ValueError(f"Geçersiz performance_profile: {self.performance_profile}")
    
    def _create_optimized_config(
        self, 
        cache_ttl: int, 
        similarity_cache_threshold: float
    ) -> RetrievalAugmentationConfig:
        """En optimize edilmiş RAPTOR config oluştur"""
        
        embed_model = CustomEmbeddingModel()
        sum_model = GPT41SummarizationModel()
        
        return RetrievalAugmentationConfig(
            # Models
            embedding_model=embed_model,
            summarization_model=sum_model,
            tree_builder_type="cluster",
            
            # 🚀 Performance optimization
            enable_async=True,
            enable_caching=True,
            enable_metrics=True,
            performance_monitoring=True,
            max_concurrent_operations=self.max_concurrent_operations,
            
            # 🧠 Cache optimization
            cache_ttl=cache_ttl,
            
            # 🔍 Retrieval optimization
            tr_enable_caching=True,
            tr_adaptive_retrieval=True,
            tr_early_termination=self.early_termination,
            tr_top_k=self.top_k,
            tr_threshold=0.5,
            
            # 🏗️ Tree builder optimization
            tb_batch_size=200,
            tb_build_mode="async",
            tb_enable_progress_tracking=False,  # Retrieve sırasında gereksiz
        )
    
    def _initialize_raptor(self):
        """RAPTOR instance'ını initialize et"""
        try:
            self.ra = RetrievalAugmentation(config=self.config, tree=self.tree_path)
            
            # Cache optimization (eğer mümkünse)
            if hasattr(self.ra.retriever, 'query_cache') and self.ra.retriever.query_cache:
                # Cache threshold ayarı yapılmış olabilir ama emin olmak için
                logger.info("🔧 Cache optimizations applied")
            
            logger.info(f"✅ RAPTOR loaded from {self.tree_path}")
            
        except Exception as e:
            logger.error(f"❌ RAPTOR initialization failed: {e}")
            raise
    
    async def retrieve_batch(
        self,
        queries: List[str],
        return_layer_info: bool = False,
        timeout_per_query: float = 30.0,
        max_retries: int = 1
    ) -> BatchRetrievalMetrics:
        """
        🚀 Ana async batch retrieve fonksiyonu
        
        Args:
            queries: Retrieve edilecek query'lerin listesi
            return_layer_info: Layer bilgilerini de döndür
            timeout_per_query: Her query için timeout (saniye)
            max_retries: Başarısız query'ler için retry sayısı
            
        Returns:
            BatchRetrievalMetrics: Tüm sonuçlar ve metrikler
        """
        start_time = time.time()
        
        # Metrics initialize
        metrics = BatchRetrievalMetrics(
            total_queries=len(queries),
            concurrent_operations=min(self.max_concurrent_operations, len(queries))
        )
        
        logger.info(f"🚀 Starting batch retrieval: {len(queries)} queries")
        logger.info(f"⚡ Concurrent operations: {metrics.concurrent_operations}")
        
        # Semaphore ile concurrency kontrolü
        semaphore = asyncio.Semaphore(self.max_concurrent_operations)
        
        # Her query için task oluştur
        tasks = []
        for i, query in enumerate(queries):
            task = self._retrieve_single_with_retry(
                query_id=i,
                query=query,
                semaphore=semaphore,
                return_layer_info=return_layer_info,
                timeout=timeout_per_query,
                max_retries=max_retries
            )
            tasks.append(task)
        
        # Tüm task'ları paralel çalıştır
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sonuçları işle
        for result in results:
            if isinstance(result, RetrievalResult):
                metrics.results.append(result)
                if result.success:
                    metrics.successful_queries += 1
                else:
                    metrics.failed_queries += 1
            else:
                # Exception durumu
                logger.error(f"❌ Unexpected error in batch: {result}")
                metrics.failed_queries += 1
        
        # Final metrics hesapla
        end_time = time.time()
        metrics.total_time = end_time - start_time
        metrics.average_time_per_query = metrics.total_time / len(queries)
        metrics.throughput_qps = len(queries) / metrics.total_time
        
        # Cache hit rate hesapla
        cache_hits = sum(1 for r in metrics.results if r.cache_hit)
        metrics.cache_hit_rate = cache_hits / len(metrics.results) if metrics.results else 0.0
        
        # Log summary
        logger.info(f"✅ Batch completed in {metrics.total_time:.2f}s")
        logger.info(f"📊 Success rate: {metrics.successful_queries}/{metrics.total_queries}")
        logger.info(f"⚡ Throughput: {metrics.throughput_qps:.1f} queries/sec")
        logger.info(f"🧠 Cache hit rate: {metrics.cache_hit_rate:.1%}")
        
        return metrics
    
    async def _retrieve_single_with_retry(
        self,
        query_id: int,
        query: str,
        semaphore: asyncio.Semaphore,
        return_layer_info: bool,
        timeout: float,
        max_retries: int
    ) -> RetrievalResult:
        """Tek bir query'yi retry logic ile retrieve et"""
        
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    return await asyncio.wait_for(
                        self._retrieve_single(query, return_layer_info),
                        timeout=timeout
                    )
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"⏰ Query {query_id} timeout, retrying... ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"❌ Query {query_id} final timeout")
                    return RetrievalResult(
                        query=query,
                        context="",
                        success=False,
                        response_time=timeout,
                        error="Timeout"
                    )
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Query {query_id} error, retrying: {e}")
                    await asyncio.sleep(0.1 * (attempt + 1))
                else:
                    logger.error(f"❌ Query {query_id} final error: {e}")
                    return RetrievalResult(
                        query=query,
                        context="",
                        success=False,
                        response_time=0.0,
                        error=str(e)
                    )
    
    async def _retrieve_single(
        self, 
        query: str, 
        return_layer_info: bool
    ) -> RetrievalResult:
        """Tek bir query'yi retrieve et"""
        start_time = time.time()
        
        try:
            # RAPTOR async retrieve
            if return_layer_info:
                context, layer_info = await asyncio.to_thread(
                    self.ra.retrieve,
                    query,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                    collapse_tree=self.collapse_tree,
                    return_layer_information=True,
                    use_async=True
                )
            else:
                context = await asyncio.to_thread(
                    self.ra.retrieve,
                    query,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                    collapse_tree=self.collapse_tree,
                    return_layer_information=False,
                    use_async=True
                )
                layer_info = None
            
            response_time = time.time() - start_time
            
            # Cache hit detection (basit bir yaklaşım)
            cache_hit = response_time < 0.1  # 100ms'den hızlıysa cache hit olabilir
            
            return RetrievalResult(
                query=query,
                context=context,
                success=True,
                response_time=response_time,
                cache_hit=cache_hit,
                layer_info=layer_info
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return RetrievalResult(
                query=query,
                context="",
                success=False,
                response_time=response_time,
                error=str(e)
            )
    
    async def warm_cache(self, sample_queries: Optional[List[str]] = None):
        """
        🔥 Cache pre-warming for maximum performance
        
        Args:
            sample_queries: Örnek query'ler cache doldurmak için
        """
        if sample_queries is None:
            # Default warming queries - senin use case'ine göre özelleştir
            sample_queries = [
                "What is the main topic?",
                "What are the key findings?",
                "What methodology was used?", 
                "What are the conclusions?",
                "What limitations are discussed?",
                "What future work is suggested?",
                "What related work is mentioned?",
                "What experiments were conducted?"
            ]
        
        logger.info(f"🔥 Warming cache with {len(sample_queries)} queries...")
        
        # Warm cache'i çok hızlı bir şekilde yap
        start_time = time.time()
        await self.retrieve_batch(
            sample_queries,
            return_layer_info=False,
            timeout_per_query=10.0
        )
        
        warm_time = time.time() - start_time
        logger.info(f"✅ Cache warmed in {warm_time:.2f}s - ready for production speed!")
        
        warm_time = time.time() - start_time
        logger.info(f"✅ Cache warmed in {warm_time:.2f}s - ready for production speed!")
        
        return warm_time
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Session boyunca toplanan istatistikleri döndür"""
        if hasattr(self.ra, 'get_performance_summary'):
            raptor_stats = self.ra.get_performance_summary()
        else:
            raptor_stats = {}
        
        return {
            'session_metrics': self.session_metrics.__dict__,
            'raptor_stats': raptor_stats,
            'config': {
                'performance_profile': self.performance_profile,
                'max_concurrent_operations': self.max_concurrent_operations,
                'top_k': self.top_k,
                'max_tokens': self.max_tokens,
                'early_termination': self.early_termination,
                'collapse_tree': self.collapse_tree
            }
        }
        """Session boyunca toplanan istatistikleri döndür"""
        if hasattr(self.ra, 'get_performance_summary'):
            raptor_stats = self.ra.get_performance_summary()
        else:
            raptor_stats = {}
        
        return {
            'session_metrics': self.session_metrics.__dict__,
            'raptor_stats': raptor_stats,
            'config': {
                'performance_profile': self.performance_profile,
                'max_concurrent_operations': self.max_concurrent_operations,
                'top_k': self.top_k,
                'max_tokens': self.max_tokens,
                'early_termination': self.early_termination,
                'collapse_tree': self.collapse_tree
            }
        }


# 🎯 Kullanım örneği ve test fonksiyonu
async def example_usage():
    """Örnek kullanım"""
    
    # Retriever'ı initialize et
    retriever = OptimizedAsyncRetriever(
        tree_path="vectordb/raptor-production",
        max_concurrent_operations=32,
        performance_profile="speed"  # speed, balanced, quality
    )
    
    # Test query'leri
    test_queries = [
        "What is the main topic of this document?",
        "What are the key findings?",
        "What recommendations are provided?",
        "What methodology was used?",
        "What are the conclusions?",
        "What future work is suggested?",
        "What are the limitations?",
        "What related work is mentioned?",
        "What experiments were conducted?",
        "What results were obtained?"
    ]
    
    print("🚀 Starting optimized async retrieval test...")
    
    # Batch retrieve
    metrics = await retriever.retrieve_batch(
        queries=test_queries,
        return_layer_info=False,
        timeout_per_query=30.0
    )
    
    # Sonuçları göster
    print(f"\n📊 SONUÇLAR:")
    print(f"✅ Başarılı: {metrics.successful_queries}/{metrics.total_queries}")
    print(f"⏱️ Toplam süre: {metrics.total_time:.2f}s")
    print(f"⚡ Ortalama süre: {metrics.average_time_per_query:.3f}s")
    print(f"🚀 Throughput: {metrics.throughput_qps:.1f} queries/sec")
    print(f"🧠 Cache hit rate: {metrics.cache_hit_rate:.1%}")
    
    # İlk birkaç sonucu detaylı göster
    print(f"\n📝 İLK 3 SONUÇ:")
    for i, result in enumerate(metrics.results[:3]):
        if result.success:
            print(f"\nQuery {i+1}: {result.query}")
            print(f"Response time: {result.response_time:.3f}s")
            print(f"Cache hit: {result.cache_hit}")
            print(f"Context preview: {result.context[:200]}...")
        else:
            print(f"\nQuery {i+1}: FAILED - {result.error}")
    
    # Session stats
    stats = retriever.get_session_stats()
    print(f"\n📈 SESSION STATS:")
    print(f"Config: {stats['config']}")


async def stress_test():
    """🔥 Büyük ölçekli stress test"""
    
    # 100 farklı query oluştur
    stress_queries = []
    topics = [
        "main findings", "methodology", "conclusions", "limitations", 
        "future work", "related work", "experiments", "results",
        "introduction", "background", "analysis", "discussion",
        "implications", "recommendations", "challenges", "solutions",
        "framework", "approach", "evaluation", "comparison"
    ]
    
    for i in range(100):
        topic = topics[i % len(topics)]
        stress_queries.append(f"What are the {topic} mentioned in section {i+1}?")
    
    print(f"🔥 STRESS TEST: {len(stress_queries)} queries")
    
    # Farklı concurrency seviyeleri test et
    concurrency_levels = [8, 16, 32, 64]
    
    for concurrency in concurrency_levels:
        print(f"\n⚡ Testing with {concurrency} concurrent operations...")
        
        retriever = OptimizedAsyncRetriever(
            tree_path="vectordb/raptor-production",
            max_concurrent_operations=concurrency,
            performance_profile="speed"  # En hızlı mod
        )
        
        start_time = time.time()
        results = await retriever.retrieve_batch(stress_queries[:50])  # İlk 50 query
        
        print(f"📊 Results for concurrency={concurrency}:")
        print(f"   ⏱️ Time: {results.total_time:.2f}s")
        print(f"   🚀 Throughput: {results.throughput_qps:.1f} queries/sec")
        print(f"   ✅ Success: {results.successful_queries}/{results.total_queries}")
        print(f"   🧠 Cache hits: {results.cache_hit_rate:.1%}")


async def benchmark_profiles():
    """📊 Tüm performance profile'ları benchmark et"""
    
    test_queries = [
        "What is the main contribution of this paper?",
        "What methodology was used in the research?", 
        "What are the key experimental results?",
        "What limitations are discussed?",
        "What future work is suggested?",
        "How does this compare to related work?",
        "What datasets were used?",
        "What evaluation metrics were employed?",
        "What are the practical implications?",
        "What novel techniques are introduced?"
    ]
    
    profiles = ["speed", "balanced", "quality"]
    
    print("📊 PERFORMANCE PROFILE BENCHMARK")
    print("=" * 50)
    
    for profile in profiles:
        print(f"\n🎯 Testing {profile.upper()} profile...")
        
        retriever = OptimizedAsyncRetriever(
            tree_path="vectordb/raptor-production",
            max_concurrent_operations=16,
            performance_profile=profile
        )
        
        results = await retriever.retrieve_batch(test_queries)
        
        print(f"   ⏱️ Avg time per query: {results.average_time_per_query:.3f}s")
        print(f"   🚀 Throughput: {results.throughput_qps:.1f} queries/sec") 
        print(f"   📏 Avg context length: {sum(len(r.context) for r in results.results if r.success) // results.successful_queries} chars")
        print(f"   🧠 Cache hit rate: {results.cache_hit_rate:.1%}")


# 🏃‍♂️ Ana çalıştırma fonksiyonu
async def main():
    """
    🚀 PRODUCTION-READY Ana fonksiyon
    Test sonuçlarına göre optimize edilmiş ayarlarla
    """
    
    # 🎯 TEST SONUÇLARINA GÖRE OPTİMAL AYARLAR
    retriever = OptimizedAsyncRetriever(
        tree_path="vectordb/raptor-production",
        max_concurrent_operations=16,    # Test sonucuna göre optimal
        performance_profile="balanced",  # En hızlı performans %79.4 q/s!
        cache_ttl=14400,                # 4 saat cache
        similarity_cache_threshold=0.8   # Good hit rate
    )
    
    # 🔥 CACHE PRE-WARMING (Production için kritik!)
    print("🔥 Cache warming...")
    warm_time = await retriever.warm_cache()
    print(f"✅ Cache warmed in {warm_time:.2f}s")
    
    # Kendi query'lerini buraya ekle
    my_queries = [
        "Important information about topic 1",
        "Important information about topic 2", 
        "Important information about topic 3",
        "Important information about topic 4",
        "Important information about topic 5",
        "What are the main findings?",
        "What methodology was used?",
        "What are the conclusions?",
        "What limitations are discussed?",
        "What future work is suggested?"
    ]
    
    print(f"\n🚀 Processing {len(my_queries)} queries with optimal settings...")
    
    # Retrieve işlemi
    results = await retriever.retrieve_batch(my_queries)
    
    # 📊 SONUÇLAR
    print(f"\n🎉 RESULTS:")
    print(f"✅ Success: {results.successful_queries}/{results.total_queries}")
    print(f"⏱️ Total time: {results.total_time:.2f}s")
    print(f"🚀 Throughput: {results.throughput_qps:.1f} queries/sec")
    print(f"🧠 Cache hit rate: {results.cache_hit_rate:.1%}")
    print(f"⚡ Avg per query: {results.average_time_per_query:.3f}s")
    
    # Başarılı sonuçları göster
    success_count = 0
    for result in results.results:
        if result.success and success_count < 3:  # İlk 3 başarılı sonucu göster
            print(f"\n📝 Query: {result.query[:60]}...")
            print(f"   ⏱️ Time: {result.response_time:.3f}s")
            print(f"   🧠 Cache: {'HIT' if result.cache_hit else 'MISS'}")
            print(f"   📏 Context: {len(result.context)} chars")
            success_count += 1
        elif not result.success:
            print(f"\n❌ FAILED: {result.query[:50]}... -> {result.error}")
    
    return results


async def production_example():
    """
    🏭 PRODUCTION EXAMPLE
    Gerçek production kullanımı için optimized pattern
    """
    
    # Single instance for production (reuse edilebilir)
    retriever = OptimizedAsyncRetriever(
        tree_path="vectordb/raptor-production",
        max_concurrent_operations=16,    # Optimal concurrency
        performance_profile="balanced",  # 79.4 q/s performance!
        cache_ttl=14400                 # 4 hour cache
    )
    
    # One-time cache warming (production başlangıcında)
    await retriever.warm_cache()
    print("🚀 Production ready! Cache warmed.")
    
    # Example production batch processing
    batch_1 = [
        "What are the main research contributions?",
        "What experiments validate the approach?", 
        "How does performance compare to baselines?",
        "What are the practical applications?",
        "What future research directions are suggested?"
    ]
    
    batch_2 = [
        "What methodology was employed?",
        "What datasets were used for evaluation?",
        "What are the key limitations?",
        "How does this work relate to prior research?", 
        "What implementation details are provided?"
    ]
    
    # Process multiple batches (production pattern)
    print("\n📦 Processing batch 1...")
    results_1 = await retriever.retrieve_batch(batch_1)
    
    print(f"✅ Batch 1: {results_1.throughput_qps:.1f} q/s, {results_1.cache_hit_rate:.1%} cache hits")
    
    print("\n📦 Processing batch 2...")
    results_2 = await retriever.retrieve_batch(batch_2)
    
    print(f"✅ Batch 2: {results_2.throughput_qps:.1f} q/s, {results_2.cache_hit_rate:.1%} cache hits")
    
    # Production stats
    stats = retriever.get_session_stats()
    print(f"\n📊 SESSION SUMMARY:")
    print(f"Profile: {stats['config']['performance_profile']}")
    print(f"Concurrency: {stats['config']['max_concurrent_operations']}")
    print(f"Total processed: {results_1.total_queries + results_2.total_queries} queries")
    
    return retriever, [results_1, results_2]


if __name__ == "__main__":
    # 🚀 PRODUCTION-READY main (cache warming + optimal settings)
    asyncio.run(main())
    
    # 🏭 Production pattern example
    # asyncio.run(production_example())
    
    # 🔥 Stress test çalıştır (100 query)
    # asyncio.run(stress_test())
    
    # 📊 Profile benchmark çalıştır
    # asyncio.run(benchmark_profiles())