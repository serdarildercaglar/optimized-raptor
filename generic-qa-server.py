"""
DOSYA: generic-qa-server.py
AÇIKLAMA: Jenerik soru-cevap asistanı - RAPTOR RAG sistemi ile entegre WebSocket sunucusu
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

app = FastAPI(title="Generic QA Server", description="RAPTOR-powered Question Answering System")

# Redis yapılandırması
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    password='Ph4nt0m4+4',
    decode_responses=True
)

openai_client = AsyncOpenAI()

# WebSocket bağlantı yöneticisi
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def send_stream_chunk(self, chunk: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(chunk))

manager = ConnectionManager()

# 🚀 PARALEL RETRIEVE FONKSİYONU
async def batch_retrieve_async(queries: List[str]) -> Dict[str, any]:
    """
    Paralel retrieve + unique content filtering
    Birden fazla sorguyu aynı anda işler ve duplicate içerikleri kaldırır
    """
    if not queries:
        return {"unique_contexts": [], "query_mapping": {}, "stats": {}}
    
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

async def process_stream_response(response_stream, session_id: str, client_id: str):
    """
    OpenAI stream response'unu işler ve tool calls varsa onları yönetir
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
                print(f"JSON decode error for arguments: {tool_call['function']['arguments']}")
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
            print(f"Final response error: {e}")
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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Ana WebSocket endpoint - kullanıcı ile etkileşim"""
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
                print(f"OpenAI API error: {e}")
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
        print(f"WebSocket error: {e}")
        manager.disconnect(client_id)

# Chat geçmişini getiren endpoint
@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    """Belirli bir session'ın chat geçmişini döndürür"""
    history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
    return [json.loads(msg) for msg in history]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü"""
    try:
        # Redis bağlantısını test et
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # RAPTOR durumunu kontrol et
    raptor_status = "healthy" if RA and RA.tree else "unhealthy"
    
    return {
        "status": "healthy" if redis_status == "healthy" and raptor_status == "healthy" else "unhealthy",
        "redis": redis_status,
        "raptor": raptor_status,
        "active_connections": len(manager.active_connections)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )