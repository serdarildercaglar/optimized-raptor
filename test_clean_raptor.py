# test_clean_raptor.py - TEMİZ RAPTOR TEST SCRIPT'İ
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
        
        print("\n🔍 RETRIEVAL TESTLER")
        print("-" * 30)
        
        for method in ["dense", "sparse", "hybrid"]:
            print(f"\n📋 {method.upper()} Method:")
            
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
        
        print("\n🤖 QA TESTLER")  
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
                
                print(f"\n{i}. Soru: {question}")
                print(f"   Cevap: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                print(f"   ⏱️ {qa_time:.3f}s")
                
            except Exception as e:
                print(f"\n{i}. Soru: {question} - ❌ HATA: {e}")
        
        # Tree'yi kaydet
        print("\n💾 Tree kaydediliyor...")
        raptor.save("clean_raptor_tree.pkl")
        print("✅ Tree kaydedildi: clean_raptor_tree.pkl")
        
        # Tree'yi yükle
        print("\n📂 Tree yükleniyor...")
        new_raptor = CleanRAPTOR()
        new_raptor.load("clean_raptor_tree.pkl")
        
        # Yüklenen tree ile test
        test_context = new_raptor.retrieve("yapay zeka", method="hybrid")
        print(f"✅ Yüklenen tree test edildi: {len(test_context)} karakter")
        
        print("\n" + "=" * 40)
        print("🎉 TEMİZ RAPTOR BAŞARIYLA TEST EDİLDİ!")
        print("🚀 Sistem kullanıma hazır!")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clean_raptor()
