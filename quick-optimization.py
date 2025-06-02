# quick_optimization.py - Enhanced RAPTOR Hızlı Optimizasyon
"""
5 kritik optimizasyon ile hybrid retrieval hızını 5.4s'den ~1s'ye düşür
"""

import asyncio
import time
from pathlib import Path

# Enhanced RAPTOR imports
from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, 
    HybridConfig, 
    create_enhanced_raptor
)
from raptor.hybrid_retriever import FusionMethod

print("⚡ Enhanced RAPTOR Hızlı Optimizasyon Başlıyor...")

# 1️⃣ OPTIMIZE EDİLMİŞ HYBRID CONFIG
def create_optimized_hybrid_config():
    """Hız odaklı hybrid konfigürasyonu"""
    return HybridConfig(
        enable_hybrid=True,
        enable_query_enhancement=True,  # Sadece gerektiğinde
        enable_sparse_retrieval=True,
        enable_reranking=False,  # 🚀 DISABLE RERANKING (Major speedup)
        
        # Fusion settings - RRF fastest method
        fusion_method=FusionMethod.RRF,  # En hızlı fusion method
        dense_weight=0.7,  # Dense'e ağırlık ver (daha hızlı)
        sparse_weight=0.3,
        
        # Sparse retrieval settings - optimized
        sparse_algorithm="bm25_okapi",
        sparse_k1=1.5,  # 🚀 Slightly higher for better precision
        sparse_b=0.6,   # 🚀 Lower for speed
        
        # Query enhancement settings - reduced
        max_query_expansions=3,  # 🚀 5'ten 3'e düşür (speed boost)
        semantic_expansion=False,  # 🚀 DISABLE semantic expansion (major speedup)
        
        # Reranking settings - disabled for speed
        rerank_top_k=0,  # Disabled
        
        # Performance settings
        enable_caching=True,
        cache_dir="hybrid_cache_optimized"
    )

# 2️⃣ OPTIMIZE EDİLMİŞ RA CONFIG  
def create_optimized_ra_config():
    """Hız odaklı RA konfigürasyonu"""
    embed_model = CustomEmbeddingModel()
    sum_model = GPT41SummarizationModel()
    
    return RetrievalAugmentationConfig(
        # Tree Builder optimizations
        tb_summarization_length=256,  # Balanced
        tb_max_tokens=100,
        tb_num_layers=3,  # 🚀 Reduced from 5 to 3 for speed
        tb_batch_size=200,  # 🚀 Increased batch size
        tb_build_mode="async",
        tb_enable_progress_tracking=False,  # 🚀 Disable for speed
        tb_threshold=0.4,  # 🚀 Higher threshold for faster clustering
        tb_top_k=6,  # 🚀 Reduced from 8 to 6
        
        # Tree Retriever optimizations  
        tr_enable_caching=True,
        tr_adaptive_retrieval=False,  # 🚀 DISABLE adaptive (speedup)
        tr_early_termination=True,
        tr_threshold=0.5,
        tr_top_k=8,  # 🚀 Balanced
        
        # Enhanced pipeline features
        enable_async=True,
        enable_caching=True,
        enable_metrics=False,  # 🚀 DISABLE metrics for speed
        enable_progress_tracking=False,  # 🚀 DISABLE for speed
        performance_monitoring=False,  # 🚀 DISABLE for speed
        max_concurrent_operations=20,  # 🚀 Increased concurrency
        cache_ttl=10800,  # 3 hours
        
        # Models
        summarization_model=sum_model, 
        embedding_model=embed_model,
        tree_builder_type="cluster",
    )

# 3️⃣ CACHE WARMING FUNCTION
async def warm_cache(enhanced_RA):
    """Cache'i önceden doldur"""
    print("🔥 Cache warming başlıyor...")
    
    warming_queries = [
        "doküman konusu",
        "ana başlık", 
        "önemli bilgi",
        "temel kavram",
        "ana fikir"
    ]
    
    start_time = time.time()
    
    # Her method için cache warming
    for method in ["dense", "sparse", "hybrid"]:
        for query in warming_queries:
            try:
                await asyncio.to_thread(
                    enhanced_RA.retrieve_enhanced,
                    query, 
                    method=method, 
                    top_k=5, 
                    max_tokens=1000
                )
            except Exception as e:
                print(f"Cache warming error for {method}: {e}")
    
    warming_time = time.time() - start_time
    print(f"✅ Cache warming tamamlandı: {warming_time:.2f}s")

# 4️⃣ PERFORMANCE BENCHMARK
async def benchmark_performance(enhanced_RA):
    """Optimizasyon sonrası performance test"""
    print("\n📊 Performance benchmark başlıyor...")
    
    test_queries = [
        "Bu dokümanın ana konusu nedir?",
        "Önemli başlıklar",
        "Ana fikirler nelerdir?",
        "Temel kavramlar",
        "Doküman özeti"
    ]
    
    results = {}
    
    for method in ["dense", "sparse", "hybrid"]:
        print(f"\n🔄 {method.upper()} method test ediliyor...")
        
        method_times = []
        success_count = 0
        
        for i, query in enumerate(test_queries, 1):
            try:
                start_time = time.time()
                
                result = await asyncio.to_thread(
                    enhanced_RA.retrieve_enhanced,
                    query, 
                    method=method, 
                    top_k=6,
                    max_tokens=1500,
                    enhance_query=(method == "hybrid"),  # Sadece hybrid için enhancement
                    return_detailed=False
                )
                
                elapsed = time.time() - start_time
                method_times.append(elapsed)
                success_count += 1
                
                print(f"   Query {i}: {elapsed:.3f}s - {len(result)} chars")
                
            except Exception as e:
                print(f"   Query {i}: FAILED - {e}")
        
        if method_times:
            avg_time = sum(method_times) / len(method_times)
            success_rate = success_count / len(test_queries)
            
            results[method] = {
                'avg_time': avg_time,
                'success_rate': success_rate,
                'total_queries': len(test_queries),
                'improvement': "N/A"
            }
            
            # Improvement calculation
            if method == "hybrid":
                old_time = 5.4  # Previous benchmark
                improvement = ((old_time - avg_time) / old_time) * 100
                results[method]['improvement'] = f"{improvement:.1f}%"
            
            print(f"   📈 {method.upper()} Summary: {avg_time:.3f}s avg, {success_rate:.1%} success")
            if method == "hybrid":
                print(f"   🚀 Hybrid Speed Improvement: {results[method]['improvement']}")
    
    return results

# 5️⃣ CACHE EFFICIENCY TEST
async def test_cache_efficiency(enhanced_RA):
    """Cache efficiency test"""
    print("\n💾 Cache efficiency test başlıyor...")
    
    test_query = "Bu dokümanın ana konusu nedir?"
    
    # First call (cache miss)
    start_time = time.time()
    first_result = await asyncio.to_thread(
        enhanced_RA.retrieve_enhanced,
        test_query, 
        method="hybrid", 
        top_k=5, 
        max_tokens=1500
    )
    first_time = time.time() - start_time
    
    # Second call (should be cache hit)
    start_time = time.time()
    second_result = await asyncio.to_thread(
        enhanced_RA.retrieve_enhanced,
        test_query, 
        method="hybrid", 
        top_k=5, 
        max_tokens=1500
    )
    second_time = time.time() - start_time
    
    # Cache analysis
    if second_time < first_time:
        improvement = ((first_time - second_time) / first_time) * 100
        print(f"   📊 Cache Performance:")
        print(f"   First call (miss): {first_time:.3f}s")
        print(f"   Second call (hit): {second_time:.3f}s")
        print(f"   🚀 Cache improvement: {improvement:.1f}%")
    else:
        print(f"   ⚠️ Cache may not be working optimally")
        print(f"   First: {first_time:.3f}s, Second: {second_time:.3f}s")

# 6️⃣ MAIN OPTIMIZATION FUNCTION
async def main_optimization():
    """Ana optimizasyon fonksiyonu"""
    
    print("🎯 Enhanced RAPTOR Hızlı Optimizasyon")
    print("=" * 50)
    
    # Load existing tree with optimized configs
    PATH = "vectordb/enhanced-raptor-optimized"
    
    try:
        print("📂 Optimized konfigürasyon ile RA yükleniyor...")
        
        # Create optimized configs
        ra_config = create_optimized_ra_config()
        hybrid_config = create_optimized_hybrid_config()
        
        # Load Enhanced RAPTOR
        enhanced_RA = EnhancedRetrievalAugmentation(
            config=ra_config,
            tree=PATH,
            hybrid_config=hybrid_config
        )
        
        print("✅ Enhanced RAPTOR optimized konfigürasyon ile yüklendi!")
        
        # Step 1: Cache warming
        await warm_cache(enhanced_RA)
        
        # Step 2: Performance benchmark
        benchmark_results = await benchmark_performance(enhanced_RA)
        
        # Step 3: Cache efficiency test
        await test_cache_efficiency(enhanced_RA)
        
        # Step 4: Final summary
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZASYON SONUÇLARI")
        print("=" * 60)
        
        for method, stats in benchmark_results.items():
            improvement = stats.get('improvement', 'N/A')
            print(f"📊 {method.upper()}:")
            print(f"   Ortalama süre: {stats['avg_time']:.3f}s")
            print(f"   Başarı oranı: {stats['success_rate']:.1%}")
            if improvement != 'N/A':
                print(f"   🚀 İyileştirme: {improvement}")
            print()
        
        # Recommendations
        print("💡 SONRAKI ADIMLAR:")
        
        hybrid_time = benchmark_results.get('hybrid', {}).get('avg_time', 999)
        if hybrid_time < 2.0:
            print("   ✅ Hybrid retrieval hızı optimal seviyede!")
            print("   🚀 API test'e geçmeye hazır")
        elif hybrid_time < 3.0:
            print("   ⚡ Hybrid retrieval iyi seviyede")
            print("   🔧 Ek optimizasyon seçenekleri mevcut")
        else:
            print("   ⚠️ Hybrid retrieval daha fazla optimizasyon gerektirebilir")
            print("   🔧 Deep optimization önerilir")
        
        return enhanced_RA, benchmark_results
        
    except Exception as e:
        print(f"❌ Optimizasyon hatası: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 7️⃣ RUN OPTIMIZATION
if __name__ == "__main__":
    print("⚡ Enhanced RAPTOR Hızlı Optimizasyon başlatılıyor...")
    
    # Set up async environment
    try:
        enhanced_RA, results = asyncio.run(main_optimization())
        
        if enhanced_RA and results:
            print("\n🎯 Optimizasyon başarıyla tamamlandı!")
            print("💡 Sonraki adım için enhanced_test.py'yi çalıştırabilirsiniz")
        else:
            print("\n❌ Optimizasyon tamamlanamadı")
            
    except KeyboardInterrupt:
        print("\n⏹️ Optimizasyon kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")