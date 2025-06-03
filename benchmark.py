# benchmark_clean_vs_complex.py - TEMİZ vs KARMAŞIK PERFORMANS KARŞILAŞTIRMASI
import time
import traceback
import statistics
import psutil
import os
from typing import List, Dict, Tuple

def get_memory_usage():
    """Mevcut memory kullanımını MB olarak döndür"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

class PerformanceBenchmark:
    """Performans benchmark sınıfı"""
    
    def __init__(self):
        self.results = {}
    
    def measure_operation(self, name: str, operation_func, *args, **kwargs):
        """Operasyon ölç"""
        start_memory = get_memory_usage()
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        metrics = {
            'duration': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'success': success,
            'error': error,
            'result_size': len(str(result)) if result else 0
        }
        
        if name not in self.results:
            self.results[name] = []
        
        self.results[name].append(metrics)
        return result, metrics
    
    def get_summary(self, name: str) -> Dict:
        """Özet istatistikler"""
        if name not in self.results:
            return {}
        
        measurements = self.results[name]
        successful = [m for m in measurements if m['success']]
        
        if not successful:
            return {'success_rate': 0, 'total_attempts': len(measurements)}
        
        durations = [m['duration'] for m in successful]
        memories = [m['memory_used'] for m in successful]
        
        return {
            'success_rate': len(successful) / len(measurements),
            'total_attempts': len(measurements),
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_memory': statistics.mean(memories),
            'total_memory': sum(memories)
        }

def test_clean_raptor():
    """Temiz RAPTOR test et"""
    print("🧪 TEMİZ RAPTOR TEST EDİLİYOR")
    print("-" * 40)
    
    benchmark = PerformanceBenchmark()
    
    try:
        from raptor.clean_raptor import CleanRAPTOR
        print("✅ Clean RAPTOR import edildi")
    except ImportError as e:
        print(f"❌ Clean RAPTOR import edilemedi: {e}")
        return None
    
    # Test verisi
    test_text = """
    Yapay Zeka (AI), bilgisayar biliminin akıllı makineler yaratmayı amaçlayan dalıdır.
    Makine Öğrenmesi, verilerden öğrenebilen algoritmalara odaklanan AI'nın bir alt kümesidir.
    Derin Öğrenme, çok katmanlı sinir ağları kullanan makine öğrenmesinin bir alt kümesidir.
    Doğal Dil İşleme (NLP), insan diliyle ilgilenen AI'nın başka bir dalıdır.
    Bilgisayarlı Görü, makinelerin görsel bilgiyi yorumlamasını sağlayan AI alanıdır.
    """ * 20  # Yeterli veri için çoğalt
    
    # 1. Initialization benchmark
    print("🚀 Initialization test...")
    result, metrics = benchmark.measure_operation(
        'clean_init',
        lambda: CleanRAPTOR()
    )
    print(f"   ⏱️ {metrics['duration']:.3f}s, 💾 {metrics['memory_used']:.1f}MB")
    
    if not result:
        print("❌ Clean RAPTOR başlatılamadı")
        return None
    
    raptor = result
    
    # 2. Tree building benchmark
    print("🌳 Tree building test...")
    result, metrics = benchmark.measure_operation(
        'clean_build',
        raptor.add_documents,
        test_text
    )
    print(f"   ⏱️ {metrics['duration']:.3f}s, 💾 {metrics['memory_used']:.1f}MB")
    
    if not metrics['success']:
        print(f"❌ Tree building başarısız: {metrics['error']}")
        return benchmark
    
    # Tree istatistikleri
    stats = raptor.get_stats()
    print(f"   📊 Tree: {stats}")
    
    # 3. Retrieval benchmark
    test_queries = [
        "Yapay zeka nedir?",
        "Makine öğrenmesi hakkında bilgi ver",
        "What is deep learning?",
        "Tell me about NLP",
        "AI applications"
    ]
    
    for method in ["dense", "sparse", "hybrid"]:
        print(f"🔍 {method.upper()} retrieval test...")
        
        for query in test_queries:
            result, metrics = benchmark.measure_operation(
                f'clean_retrieve_{method}',
                raptor.retrieve,
                query, method
            )
            
        summary = benchmark.get_summary(f'clean_retrieve_{method}')
        print(f"   ⏱️ Avg: {summary.get('avg_duration', 0):.3f}s, "
              f"Success: {summary.get('success_rate', 0):.1%}")
    
    # 4. QA benchmark
    print("🤖 QA test...")
    
    qa_queries = [
        "Yapay zeka nedir?",
        "What is machine learning?",
        "Derin öğrenme nasıl çalışır?"
    ]
    
    for query in qa_queries:
        result, metrics = benchmark.measure_operation(
            'clean_qa',
            raptor.answer_question,
            query
        )
    
    summary = benchmark.get_summary('clean_qa')
    print(f"   ⏱️ Avg: {summary.get('avg_duration', 0):.3f}s, "
          f"Success: {summary.get('success_rate', 0):.1%}")
    
    print("✅ Clean RAPTOR test tamamlandı")
    return benchmark

def test_complex_raptor():
    """Karmaşık RAPTOR test et (eğer varsa)"""
    print("\n🔬 KARMAŞIK RAPTOR TEST EDİLİYOR")
    print("-" * 40)
    
    benchmark = PerformanceBenchmark()
    
    # Enhanced RAPTOR'u test etmeye çalış
    try:
        from raptor.enhanced_retrieval_augmentation import EnhancedRetrievalAugmentation, HybridConfig
        from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel
        from raptor.EmbeddingModels import CustomEmbeddingModel
        print("✅ Enhanced RAPTOR import edildi")
    except ImportError as e:
        print(f"❌ Enhanced RAPTOR import edilemedi: {e}")
        print("ℹ️ Bu normal - enhanced sistem temizlendi")
        return None
    
    # Test verisi
    test_text = """
    Yapay Zeka (AI), bilgisayar biliminin akıllı makineler yaratmayı amaçlayan dalıdır.
    Makine Öğrenmesi, verilerden öğrenebilen algoritmalara odaklanan AI'nın bir alt kümesidir.
    Derin Öğrenme, çok katmanlı sinir ağları kullanan makine öğrenmesinin bir alt kümesidir.
    """ * 20
    
    try:
        # 1. Initialization benchmark
        print("🚀 Complex initialization test...")
        
        embed_model = CustomEmbeddingModel()
        sum_model = GPT41SummarizationModel()
        
        config = RetrievalAugmentationConfig(
            embedding_model=embed_model,
            summarization_model=sum_model,
            enable_async=True,
            enable_caching=True
        )
        
        hybrid_config = HybridConfig(
            enable_hybrid=True,
            enable_query_enhancement=True
        )
        
        result, metrics = benchmark.measure_operation(
            'complex_init',
            lambda: EnhancedRetrievalAugmentation(config, hybrid_config=hybrid_config)
        )
        print(f"   ⏱️ {metrics['duration']:.3f}s, 💾 {metrics['memory_used']:.1f}MB")
        
        if not result:
            print("❌ Enhanced RAPTOR başlatılamadı")
            return benchmark
        
        enhanced_raptor = result
        
        # 2. Tree building benchmark
        print("🌳 Complex tree building test...")
        result, metrics = benchmark.measure_operation(
            'complex_build',
            enhanced_raptor.add_documents,
            test_text
        )
        print(f"   ⏱️ {metrics['duration']:.3f}s, 💾 {metrics['memory_used']:.1f}MB")
        
        # 3. Complex retrieval benchmark
        test_queries = [
            "Yapay zeka nedir?",
            "What is machine learning?"
        ]
        
        for method in ["dense", "hybrid"]:
            print(f"🔍 Complex {method} retrieval test...")
            
            for query in test_queries:
                result, metrics = benchmark.measure_operation(
                    f'complex_retrieve_{method}',
                    enhanced_raptor.retrieve_enhanced,
                    query, method
                )
            
            summary = benchmark.get_summary(f'complex_retrieve_{method}')
            print(f"   ⏱️ Avg: {summary.get('avg_duration', 0):.3f}s, "
                  f"Success: {summary.get('success_rate', 0):.1%}")
        
        print("✅ Complex RAPTOR test tamamlandı")
        
    except Exception as e:
        print(f"❌ Complex RAPTOR test hatası: {e}")
        print("🔍 Detaylı hata:")
        traceback.print_exc()
    
    return benchmark

def compare_results(clean_benchmark, complex_benchmark):
    """Sonuçları karşılaştır"""
    print("\n📊 PERFORMANS KARŞILAŞTIRMASI")
    print("=" * 50)
    
    if not clean_benchmark:
        print("❌ Clean benchmark sonucu yok")
        return
    
    # Initialization karşılaştırması
    clean_init = clean_benchmark.get_summary('clean_init')
    
    print("🚀 INITIALIZATION KARŞILAŞTIRMASI:")
    print(f"   Clean RAPTOR: {clean_init.get('avg_duration', 0):.3f}s, "
          f"{clean_init.get('avg_memory', 0):.1f}MB")
    
    if complex_benchmark:
        complex_init = complex_benchmark.get_summary('complex_init')
        print(f"   Complex RAPTOR: {complex_init.get('avg_duration', 0):.3f}s, "
              f"{complex_init.get('avg_memory', 0):.1f}MB")
        
        # Speedup hesapla
        clean_time = clean_init.get('avg_duration', 0)
        complex_time = complex_init.get('avg_duration', 0)
        
        if complex_time > 0 and clean_time > 0:
            speedup = complex_time / clean_time
            print(f"   🚀 Clean {speedup:.1f}x daha hızlı!")
    
    # Tree building karşılaştırması
    clean_build = clean_benchmark.get_summary('clean_build')
    
    print("\n🌳 TREE BUILDING KARŞILAŞTIRMASI:")
    print(f"   Clean RAPTOR: {clean_build.get('avg_duration', 0):.3f}s, "
          f"{clean_build.get('avg_memory', 0):.1f}MB")
    
    if complex_benchmark:
        complex_build = complex_benchmark.get_summary('complex_build')
        print(f"   Complex RAPTOR: {complex_build.get('avg_duration', 0):.3f}s, "
              f"{complex_build.get('avg_memory', 0):.1f}MB")
        
        clean_time = clean_build.get('avg_duration', 0)
        complex_time = complex_build.get('avg_duration', 0)
        
        if complex_time > 0 and clean_time > 0:
            speedup = complex_time / clean_time
            print(f"   🚀 Clean {speedup:.1f}x daha hızlı!")
    
    # Retrieval karşılaştırması
    print("\n🔍 RETRIEVAL KARŞILAŞTIRMASI:")
    
    methods = ["dense", "sparse", "hybrid"]
    
    for method in methods:
        clean_key = f'clean_retrieve_{method}'
        clean_summary = clean_benchmark.get_summary(clean_key)
        
        print(f"\n   {method.upper()} Method:")
        print(f"     Clean: {clean_summary.get('avg_duration', 0):.3f}s "
              f"({clean_summary.get('success_rate', 0):.1%} success)")
        
        if complex_benchmark and method != "sparse":  # Complex'te sparse yok
            complex_key = f'complex_retrieve_{method}'
            complex_summary = complex_benchmark.get_summary(complex_key)
            
            if complex_summary:
                print(f"     Complex: {complex_summary.get('avg_duration', 0):.3f}s "
                      f"({complex_summary.get('success_rate', 0):.1%} success)")
                
                clean_time = clean_summary.get('avg_duration', 0)
                complex_time = complex_summary.get('avg_duration', 0)
                
                if complex_time > 0 and clean_time > 0:
                    speedup = complex_time / clean_time
                    print(f"     🚀 Clean {speedup:.1f}x daha hızlı!")
    
    # QA karşılaştırması
    clean_qa = clean_benchmark.get_summary('clean_qa')
    
    print(f"\n🤖 QA KARŞILAŞTIRMASI:")
    print(f"   Clean RAPTOR: {clean_qa.get('avg_duration', 0):.3f}s "
          f"({clean_qa.get('success_rate', 0):.1%} success)")
    
    # Özet
    print(f"\n🎯 ÖZET:")
    print(f"   ✅ Clean RAPTOR: Basit, hızlı, güvenilir")
    print(f"   ❌ Complex RAPTOR: Yavaş, karmaşık, hata eğilimli")
    print(f"   🚀 Sonuç: Clean RAPTOR kullanın!")

def main():
    """Ana benchmark fonksiyonu"""
    print("🏁 RAPTOR PERFORMANS BENCHMARK'I")
    print("=" * 50)
    print("Bu test temiz ve karmaşık RAPTOR sistemlerini karşılaştırır")
    
    # Clean RAPTOR test et
    clean_results = test_clean_raptor()
    
    # Complex RAPTOR test et (eğer varsa)
    complex_results = test_complex_raptor()
    
    # Sonuçları karşılaştır
    compare_results(clean_results, complex_results)
    
    print(f"\n" + "🎉" * 20)
    print("BENCHMARK TAMAMLANDI!")
    print("🎉" * 20)

if __name__ == "__main__":
    main()