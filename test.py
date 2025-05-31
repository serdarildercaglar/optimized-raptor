

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from raptor import RetrievalAugmentation 
from raptor import RetrievalAugmentationConfig
from raptor import GPT4OSummarizationModel
from raptor import GPT4QAModel
from raptor.EmbeddingModels import CustomEmbeddingModel
embed_model = CustomEmbeddingModel()
sum_model = GPT4OSummarizationModel()
qa_model = GPT4QAModel()
RA_config = RetrievalAugmentationConfig(tb_summarization_length=100, tb_max_tokens=100, 
                                        qa_model=qa_model,
                                        summarization_model=sum_model, 
                                        embedding_model=embed_model) # we used default values for tb_summarization_length and tb_max_tokens


# load back the tree by passing it into RetrievalAugmentation
PATH = "vectordb/raptor"

RA = RetrievalAugmentation(tree=PATH, config=RA_config)





from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import redis.asyncio as redis
import uuid
from openai import AsyncOpenAI
import asyncio

app = FastAPI()
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    password='Ph4nt0m4+4',  # Docker Compose'da belirlediğimiz şifre
    decode_responses=True
)
openai_client = AsyncOpenAI()

# Aktif websocket bağlantılarını tutacak sınıf
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

manager = ConnectionManager()

async def process_response(response, session_id: str):
    chat_history = []
    
    # Tool calls varsa işle
    if tool_calls := response.choices[0].message.tool_calls:
        chat_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            call_id = tool_call.id
            
            # Her fonksiyon için gerekli işlemleri yap
            if function_name == "get_info":
                query = function_args['query']
                result = RA.answer_question(f"query: {query}")

                
                chat_history.append({
                    "role": "tool",
                    "content": json.dumps({"query": query, "answer": result}),
                    "tool_call_id": call_id
                })
            
            elif function_name == "user_info":
                ad = function_args['ad_soyad']
                tel = function_args['tel_no']
                result = f"'Sayın {ad} İletişim bilgileriniz kaydedildi. Size dönüş yapacağız.' bilgilerndirme mesajını dön" # fonkisyonun çağrıldığını varsayalım
                print("*"*50,"\nUser info fonksiyonu çağrıldı")
                chat_history.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": call_id
                })
            
            elif function_name == "direct_to_representative":
                result = f"Sayın {function_args['ad_soyad']}, temsilciye yönlendirildiniz. En kısa sürede size dönüş yapılacaktır."
                chat_history.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": call_id
                })

        # Final response'u al        
        final_response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history,
            temperature=0.7,
        )
        
        chat_message = final_response.choices[0].message.content
        chat_history.append({
            "role": "assistant",
            "content": chat_message
        })

    else:
        chat_message = response.choices[0].message.content
        chat_history.append({
            "role": "assistant", 
            "content": chat_message
        })
        
    # Redis'e kaydet
    await redis_client.rpush(f"chat_history:{session_id}", json.dumps(chat_history[-1]))
    
    return chat_message

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
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
            formatted_history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
            formatted_history = [json.loads(msg) for msg in formatted_history]            
            # OpenAI API çağrısı
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Sen, profesyonel bir müşteri destek chatbotusun. Görevlerin şunlardır:
                        1. Kullanıcının mesajlarını analiz ederek ürünler, hizmetler, paketler veya politikalar hakkında bilgi isteyip istemediğini belirle.
                        2. Ürün veya hizmetlerle ilgili bilgi talep edildiğinde, uygun bir sorgu oluştur ve `get_info` fonksiyonunu çağırarak detaylı bilgiyi al ve kullanıcıya ilet.
                        3. Konuşmanın sonunda, kullanıcıdan eğer isterse iletişim bilgilerini iste ve `user_info` fonksiyonunu ad_soyad ve tel_no parametreleriyle çağırarak bilgileri kaydet.
                        4. Her zaman kibar, profesyonel ve kullanıcı dostu bir üslup kullan.
                        5. Gereksiz tekrarlardan veya uzun açıklamalardan kaçın; mümkün olan en kısa ve net yanıtları ver.
                        6. Kesin olmayan bilgiler için tahmin yürütme; bilgi almak için yalnızca `get_info` fonksiyonunu kullan.
                        **Örnek Davranışlar:**
                        - Kullanıcı "Paketler hakkında bilgi verir misiniz?" dediğinde:
                        - `get_info("packages")` fonksiyonunu çağır.
                        - Dönüşü kullanıcıya uygun bir üslupla ilet.\n- Kullanıcı iletişim kurulmasını talep ettiğinde:
                        - "Tabii ki, sizinle iletişim kurmamız için lütfen adınızı ve telefon numaranızı paylaşır mısınız?" sorusunu sor.
                        - `user_info` fonksiyonunu ad_soyad ve tel_no parametreleriyle çağır.
                        - Amacın, kullanıcılara hızlı ve doğru bilgi sağlamak ve gerekirse iletişim kurulmasını kolaylaştırmaktır. 
                        Her zaman Türkçe yanıt ver.
                        """
                    }
                ] + formatted_history,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_info",
                            "description": "Ürünler, hizmetler ve politikalar hakkında detaylı bilgi al.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Alınacak bilgi için sorgu oluştur."
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "user_info",
                            "description": "Kullanıcı daha sonra iletişim kurmak veya yetkili birinin kullanıcıy arayabilmesi için kullanıcının iletişim bilgilerini kaydeder.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "ad_soyad": {
                                        "type": "string",
                                        "description": "Kullanıcının adı ve soyadı."
                                    },
                                    "tel_no": {
                                        "type": "string",
                                        "description": "Kullanıcının telefon numarası."
                                    }
                                },
                                "required": ["ad_soyad", "tel_no"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "direct_to_representative",
                            "description": "Seni aşan konularda temsilciye yönlendir.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "info_about_issue": {
                                        "type": "string",
                                        "description": "Temsilciye iletilmesi gereken konu hakkında bilgi."
                                    },
                                    "ad_soyad": {
                                        "type": "string",
                                        "description": "Kullanıcının adı ve soyadı."
                                    }
                                },
                                "required": ["info_about_issue", "ad_soyad"]
                            }
                        }
                    }
                ]
            )
            
            # Yanıtı işle ve gönder
            assistant_message = await process_response(response, session_id)
            await manager.send_message(assistant_message, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Geçmiş sohbeti getirmek için endpoint
@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
    return [json.loads(msg) for msg in history]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)