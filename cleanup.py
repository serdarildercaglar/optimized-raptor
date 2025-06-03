# cleanup_raptor.py - KARMAŞIK KODU TEMİZLEME SCRIPT'İ
import os
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def cleanup_raptor_mess():
    """Karmaşık RAPTOR kodlarını temizle"""
    
    print("🧹 RAPTOR CLEANUP BAŞLANIYOR")
    print("=" * 50)
    
    # 1. Gereksiz dosyaları sil
    files_to_remove = [
        "raptor/enhanced_retrieval_augmentation.py",
        "raptor/enhanced_retrieval_augmentation_optimized.py", 
        "raptor/hybrid_retriever_optimized.py",
        "raptor/query_enhancement_optimized.py",
        "raptor/cluster_tree_builder_optimized.py",
        "raptor/evaluation_framework.py",
        "raptor/sparse_retriever.py",
        "build-enhanced-raptor-full.py",
        "apply_major_optimizations.py", 
        "test_major_optimizations.py"
    ]
    
    print("🗑️ Gereksiz dosyalar siliniyor...")
    removed_count = 0
    for file_path in files_to_remove:
        if Path(file_path).exists():
            try:
                os.remove(file_path)
                print(f"   ✅ Silindi: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Silinemedi {file_path}: {e}")
        else:
            print(f"   ⚪ Zaten yok: {file_path}")
    
    print(f"📊 {removed_count} dosya silindi")
    
    # 2. Gereksiz dizinleri sil
    dirs_to_remove = [
        "backup_before_optimization",
        "enhanced_cache", 
        "optimized_hybrid_cache",
        "enhanced_hybrid_cache",
        "enhanced_evaluation_results",
        "evaluation_results"
    ]
    
    print("\n📁 Gereksiz dizinler siliniyor...")
    for dir_path in dirs_to_remove:
        if Path(dir_path).exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Silindi: {dir_path}/")
            except Exception as e:
                print(f"   ❌ Silinemedi {dir_path}: {e}")
    
    # 3. __init__.py'yi temizle
    print("\n🔧 __init__.py temizleniyor...")
    
    clean_init_content = '''# raptor/__init__.py - TEMİZ VERSİYON

# Temel bileşenler
from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel, CustomEmbeddingModel
from .FaissRetriever import FaissRetriever, FaissRetrieverConfig  
from .QAModels import BaseQAModel, GPT41QAModel, GPT4OMINIQAModel, GPT4QAModel
from .RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from .Retrievers import BaseRetriever
from .SummarizationModels import (
    BaseSummarizationModel,
    GPT4OMiniSummarizationModel, 
    GPT4OSummarizationModel,
    GPT41MiniSummarizationModel,
    GPT41SummarizationModel,
)
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

__all__ = [
    # Core classes
    'RetrievalAugmentation', 'RetrievalAugmentationConfig',
    'TreeBuilder', 'TreeBuilderConfig', 
    'TreeRetriever', 'TreeRetrieverConfig',
    'ClusterTreeBuilder', 'ClusterTreeConfig',
    'Node', 'Tree',
    
    # Models
    'BaseEmbeddingModel', 'CustomEmbeddingModel',
    'BaseQAModel', 'GPT41QAModel', 'GPT4OMINIQAModel', 'GPT4QAModel', 
    'BaseSummarizationModel', 'GPT4OMiniSummarizationModel',
    'GPT4OSummarizationModel', 'GPT41MiniSummarizationModel', 
    'GPT41SummarizationModel',
    
    # Retrievers
    'BaseRetriever', 'FaissRetriever', 'FaissRetrieverConfig'
]
'''
    
    try:
        with open("raptor/__init__.py", "w") as f:
            f.write(clean_init_content)
        print("   ✅ __init__.py temizlendi")
    except Exception as e:
        print(f"   ❌ __init__.py temizlenemedi: {e}")
    
    # 4. Temel dosyaları geri yükle (eğer backup varsa)
    print("\n↩️ Temel dosyalar kontrol ediliyor...")
    
    basic_files_to_restore = [
        "raptor/hybrid_retriever.py",
        "raptor/query_enhancement.py",
        "raptor/cluster_tree_builder.py"
    ]
    
    backup_dir = Path("backup_before_optimization")
    
    for file_path in basic_files_to_restore:
        if not Path(file_path).exists():
            backup_file = backup_dir / Path(file_path).name
            if backup_file.exists():
                try:
                    shutil.copy2(backup_file, file_path)
                    print(f"   ✅ Geri yüklendi: {file_path}")
                except Exception as e:
                    print(f"   ❌ Geri yüklenemedi {file_path}: {e}")
            else:
                print(f"   ⚠️ Backup bulunamadı: {file_path}")
        else:
            print(f"   ✅ Zaten mevcut: {file_path}")
    
    # 5. Config dosyalarını temizle
    print("\n🔧 Config dosyaları temizleniyor...")
    
    config_files = [
        "enhanced_hybrid_config.json",
        "*.optimization.json"
    ]
    
    for pattern in config_files:
        for file_path in Path(".").glob(pattern):
            try:
                file_path.unlink()
                print(f"   ✅ Silindi: {file_path}")
            except Exception as e:
                print(f"   ❌ Silinemedi {file_path}: {e}")
    
    print("\n✅ CLEANUP TAMAMLANDI!")
    print("=" * 50)
    print("🚀 Şimdi temiz RAPTOR sistemi kurulabilir.")

def install_clean_raptor():
    """Temiz RAPTOR sistemini kur"""
    
    print("\n📦 TEMİZ RAPTOR SİSTEMİ KURULUYOR")
    print("=" * 50)
    
    # Clean RAPTOR dosyasını kur
    clean_raptor_content = open("clean_raptor_system.py").read()
    
    try:
        with open("raptor/clean_raptor.py", "w") as f:
            f.write(clean_raptor_content)
        print("✅ Clean RAPTOR sistemi kuruldu: raptor/clean_raptor.py")
    except Exception as e:
        print(f"❌ Clean RAPTOR kurulamadı: {e}")
        return
    
    # Basit test dosyası oluştur
    test_content = '''# test_clean_raptor.py - TEMİZ RAPTOR TEST SCRIPT'İ
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Clean RAPTOR'u import et
try:
    from raptor.clean_raptor import CleanRAPTOR
    print("✅ Clean RAPTOR import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    exit(1)

def test_clean_raptor():
    """Temiz RAPTOR testi"""
    
    print("🧪 TEMİZ RAPTOR TEST BAŞLIYOR")
    print("=" * 40)
    
    # Test metni
    test_text = """
    Yapay Zeka (AI), bilgisayar biliminin akıllı makineler yaratmayı amaçlayan dalıdır.
    Makine Öğrenmesi, verilerden öğrenebilen algoritmalara odaklanan AI'nın bir alt kümesidir.
    Derin Öğrenme, çok katmanlı sinir ağları kullanan makine öğrenmesinin bir alt kümesidir.
    Doğal Dil İşleme (NLP), insan diliyle ilgilenen AI'nın başka bir dalıdır.
    Bilgisayarlı Görü, makinelerin görsel bilgiyi yorumlamasını sağlayan AI alanıdır.
    
    Yapay zeka teknolojileri günümüzde birçok alanda kullanılmaktadır.
    Sağlık sektöründe teşhis ve tedavi önerilerinde yardımcı olmaktadır.
    Finans sektöründe risk analizi ve fraud tespiti yapılmaktadır.
    Eğitim alanında kişiselleştirilmiş öğrenme deneyimleri sunulmaktadır.
    Ulaşım sektöründe otonom araçlar geliştirilmektedir.
    """
    
    try:
        # RAPTOR sistemini başlat
        print("🚀 Clean RAPTOR sistemi başlatılıyor...")
        raptor = CleanRAPTOR()
        
        # Tree oluştur
        print("🌳 Tree oluşturuluyor...")
        start_time = time.time()
        
        raptor.add_documents(test_text, max_tokens=100, max_layers=3)
        
        build_time = time.time() - start_time
        print(f"✅ Tree {build_time:.2f} saniyede oluşturuldu")
        
        # İstatistikleri göster
        stats = raptor.get_stats()
        print(f"📊 Tree istatistikleri: {stats}")
        
        # Test soruları
        test_queries = [
            "Yapay zeka nedir?",
            "Makine öğrenmesi hakkında bilgi ver",
            "AI hangi sektörlerde kullanılıyor?",
            "What is deep learning?",
            "Tell me about NLP"
        ]
        
        print("\\n🔍 RETRIEVAL TESTLER")
        print("-" * 30)
        
        for method in ["dense", "sparse", "hybrid"]:
            print(f"\\n📋 {method.upper()} Method:")
            
            total_time = 0
            success_count = 0
            
            for i, query in enumerate(test_queries, 1):
                try:
                    start_time = time.time()
                    context = raptor.retrieve(query, method=method, top_k=3)
                    query_time = time.time() - start_time
                    
                    total_time += query_time
                    success_count += 1
                    
                    print(f"   {i}. {query}")
                    print(f"      ⏱️ {query_time:.3f}s - {len(context)} karakter")
                    
                except Exception as e:
                    print(f"   {i}. {query} - ❌ HATA: {e}")
            
            if success_count > 0:
                avg_time = total_time / success_count
                print(f"   📈 Ortalama: {avg_time:.3f}s ({success_count}/{len(test_queries)} başarılı)")
        
        print("\\n🤖 QA TESTLER")  
        print("-" * 30)
        
        qa_queries = [
            "Yapay zeka nedir?",
            "Hangi sektörlerde AI kullanılıyor?",
            "What is machine learning?"
        ]
        
        for i, question in enumerate(qa_queries, 1):
            try:
                start_time = time.time()
                answer = raptor.answer_question(question)
                qa_time = time.time() - start_time
                
                print(f"\\n{i}. Soru: {question}")
                print(f"   Cevap: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                print(f"   ⏱️ {qa_time:.3f}s")
                
            except Exception as e:
                print(f"\\n{i}. Soru: {question} - ❌ HATA: {e}")
        
        # Tree'yi kaydet
        print("\\n💾 Tree kaydediliyor...")
        raptor.save("clean_raptor_tree.pkl")
        print("✅ Tree kaydedildi: clean_raptor_tree.pkl")
        
        # Tree'yi yükle
        print("\\n📂 Tree yükleniyor...")
        new_raptor = CleanRAPTOR()
        new_raptor.load("clean_raptor_tree.pkl")
        
        # Yüklenen tree ile test
        test_context = new_raptor.retrieve("yapay zeka", method="hybrid")
        print(f"✅ Yüklenen tree test edildi: {len(test_context)} karakter")
        
        print("\\n" + "=" * 40)
        print("🎉 TEMİZ RAPTOR BAŞARIYLA TEST EDİLDİ!")
        print("🚀 Sistem kullanıma hazır!")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clean_raptor()
'''
    
    try:
        with open("test_clean_raptor.py", "w") as f:
            f.write(test_content)
        print("✅ Test dosyası oluşturuldu: test_clean_raptor.py")
    except Exception as e:
        print(f"❌ Test dosyası oluşturulamadı: {e}")
    
    print("\n🎯 KURULUM TAMAMLANDI!")
    print("=" * 50)
    print("📝 Kullanım:")
    print("   1. python test_clean_raptor.py  # Test et")
    print("   2. from raptor.clean_raptor import CleanRAPTOR  # Kullan")

def create_simple_example():
    """Basit kullanım örneği oluştur"""
    
    example_content = '''# simple_raptor_example.py - BASİT KULLANIM ÖRNEĞİ
from raptor.clean_raptor import CleanRAPTOR
import time

def simple_example():
    """En basit RAPTOR kullanımı"""
    
    # RAPTOR oluştur
    raptor = CleanRAPTOR()
    
    # Belge ekle
    document = """
    Python bir programlama dilidir. Basit ve okunabilir söz dizimine sahiptir.
    Web geliştirme, veri bilimi ve yapay zeka projelerinde sıkça kullanılır.
    Django ve Flask gibi popüler web framework'leri vardır.
    NumPy, Pandas ve Scikit-learn gibi güçlü kütüphaneleri mevcuttur.
    """
    
    print("🔨 Tree oluşturuluyor...")
    raptor.add_documents(document)
    
    print("🔍 Soru soruluyor...")
    answer = raptor.answer_question("Python nedir?")
    
    print(f"💬 Cevap: {answer}")

if __name__ == "__main__":
    simple_example()
'''
    
    try:
        with open("simple_raptor_example.py", "w") as f:
            f.write(example_content)
        print("✅ Basit örnek oluşturuldu: simple_raptor_example.py")
    except Exception as e:
        print(f"❌ Basit örnek oluşturulamadı: {e}")

def main():
    """Ana fonksiyon"""
    try:
        # 1. Cleanup
        cleanup_raptor_mess()
        
        # 2. Clean sistem kur
        install_clean_raptor()
        
        # 3. Basit örnek oluştur  
        create_simple_example()
        
        print("\n" + "🎉" * 20)
        print("TEMİZ RAPTOR SİSTEMİ HAZIR!")
        print("🎉" * 20)
        
        print("\\n📋 SONRAKİ ADIMLAR:")
        print("   1. python test_clean_raptor.py")
        print("   2. python simple_raptor_example.py") 
        print("   3. Kendi projende kullan!")
        
    except Exception as e:
        print(f"❌ Ana hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()