# simple_raptor_example.py - BASİT KULLANIM ÖRNEĞİ
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
