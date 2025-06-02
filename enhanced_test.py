from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio
import time
import json
import uuid
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import redis.asyncio as redis
from openai import AsyncOpenAI

# Enhanced RAPTOR imports
from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel, GPT41QAModel
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, 
    HybridConfig, 
    create_enhanced_raptor
)
from raptor.hybrid_retriever import FusionMethod

print("🚀 Loading Enhanced RAPTOR with Hybrid Retrieval...")

# Initialize models
embed_model = CustomEmbeddingModel()
sum_model = GPT41SummarizationModel()
qa_model = GPT41QAModel()

# Enhanced RAPTOR Configuration
print("⚙️ Configuring Enhanced RAPTOR with hybrid features...")

# Standard RAPTOR config
ra_config = RetrievalAugmentationConfig(
    tb_summarization_length=200, 
    tb_max_tokens=100, 
    tb_num_layers=4,
    tb_batch_size=100,
    tb_build_mode="async",
    tr_enable_caching=True,
    tr_adaptive_retrieval=True,
    tr_early_termination=True,
    qa_model=qa_model,
    summarization_model=sum_model, 
    embedding_model=embed_model,
    enable_async=True,
    enable_caching=True,
    enable_metrics=True,
    performance_monitoring=True,
    max_concurrent_operations=10,
    cache_ttl=3600
)

# Hybrid configuration with advanced features
hybrid_config = HybridConfig(
    enable_hybrid=True,
    enable_query_enhancement=True,
    enable_sparse_retrieval=True,
    enable_reranking=True,
    
    # Fusion settings
    fusion_method=FusionMethod.RRF,  # Reciprocal Rank Fusion
    dense_weight=0.6,
    sparse_weight=0.4,
    
    # Sparse retrieval settings
    sparse_algorithm="bm25_okapi",
    sparse_k1=1.2,
    sparse_b=0.75,
    
    # Query enhancement settings
    max_query_expansions=5,
    semantic_expansion=True,
    
    # Reranking settings
    rerank_top_k=15,
    
    # Performance settings
    enable_caching=True,
    cache_dir="hybrid_cache"
)

# Load Enhanced RAPTOR
PATH = "vectordb/enhanced-raptor-optimized"
try:
    print(f"📂 Loading Enhanced RAPTOR from {PATH}...")
    RA = EnhancedRetrievalAugmentation(
        config=ra_config,
        tree=PATH,
        hybrid_config=hybrid_config
    )
    print("✅ Enhanced RAPTOR loaded successfully with hybrid capabilities!")
    
    # Test hybrid features
    print("\n🔍 Testing hybrid retrieval methods...")
    test_query = "Bu dokümanın ana konusu nedir?"
    
    # Test different retrieval methods
    methods = ["dense", "sparse", "hybrid"]
    for method in methods:
        start_time = time.time()
        try:
            result = RA.retrieve_enhanced(
                test_query, 
                method=method, 
                top_k=5, 
                max_tokens=1500,
                return_detailed=False
            )
            elapsed = time.time() - start_time
            print(f"   {method.upper()}: {elapsed:.3f}s - {len(result)} chars")
        except Exception as e:
            print(f"   {method.upper()}: Failed - {e}")
    
    # Show performance summary
    perf_summary = RA.get_enhanced_performance_summary()
    print(f"\n📊 Hybrid features enabled:")
    if 'hybrid_features' in perf_summary:
        features = perf_summary['hybrid_features']
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}")

except Exception as e:
    print(f"❌ Error loading Enhanced RAPTOR: {e}")
    print("💡 Make sure you've built the tree with build-enhanced-raptor.py first!")
    exit(1)

# Initialize FastAPI and other components
app = FastAPI(title="Enhanced RAPTOR API with Hybrid Retrieval")
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    password='Ph4nt0m4+4',
    decode_responses=True
)
openai_client = AsyncOpenAI()

# Connection manager for WebSocket
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

# 🚀 ENHANCED PARALLEL RETRIEVE WITH HYBRID METHODS
async def enhanced_batch_retrieve_async(queries: List[str], method: str = "hybrid") -> Dict[str, any]:
    """
    Enhanced parallel retrieve with hybrid methods + intelligent deduplication
    
    Args:
        queries: List of search queries
        method: Retrieval method ("hybrid", "dense", "sparse")
    
    Returns:
        Dictionary with unique contexts, query mapping, and performance stats
    """
    if not queries:
        return {"unique_contexts": [], "query_mapping": {}, "stats": {}, "method_used": method}
    
    start_time = time.time()
    
    # Enhanced retrieve function for single query
    async def retrieve_single_enhanced(query: str) -> tuple:
        try:
            # Use enhanced retrieval with method selection
            result = RA.retrieve_enhanced(
                query, 
                method=method,
                top_k=8,
                max_tokens=2000,
                enhance_query=True,  # Enable query enhancement
                return_detailed=True
            )
            
            if isinstance(result, tuple):
                context, detailed_results = result
                return query, context, detailed_results
            else:
                return query, result, []
                
        except Exception as e:
            return query, f"Error retrieving '{query}': {str(e)}", []
    
    # 🚀 PARALLEL EXECUTION with enhanced retrieval
    tasks = [retrieve_single_enhanced(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 🔍 INTELLIGENT CONTENT DEDUPLICATION
    seen_contents = set()
    unique_contexts = []
    query_mapping = {}
    duplicate_count = 0
    total_confidence = 0.0
    method_stats = {"dense_used": 0, "sparse_used": 0, "hybrid_used": 0}
    
    for query, result, detailed_results in results:
        if isinstance(result, Exception):
            continue
        
        # Normalize content for deduplication
        normalized_content = ' '.join(str(result).split())
        
        # Calculate content hash for better deduplication
        content_hash = hash(normalized_content[:500])  # Use first 500 chars for hash
        
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_contexts.append(str(result))
            
            # Enhanced mapping with method info
            query_mapping[len(unique_contexts) - 1] = {
                "primary_query": query,
                "method_used": method,
                "content_preview": normalized_content[:150] + "..." if len(normalized_content) > 150 else normalized_content,
                "confidence": getattr(detailed_results[0], 'confidence', 0.0) if detailed_results else 0.0,
                "hybrid_details": {
                    "dense_score": getattr(detailed_results[0], 'dense_score', 0.0) if detailed_results else 0.0,
                    "sparse_score": getattr(detailed_results[0], 'sparse_score', 0.0) if detailed_results else 0.0,
                    "fused_score": getattr(detailed_results[0], 'fused_score', 0.0) if detailed_results else 0.0,
                } if detailed_results and hasattr(detailed_results[0], 'dense_score') else {}
            }
            
            # Update confidence tracking
            if detailed_results and hasattr(detailed_results[0], 'confidence'):
                total_confidence += detailed_results[0].confidence
            
            # Track method usage
            method_stats[f"{method}_used"] += 1
            
        else:
            duplicate_count += 1
            # Find and update existing mapping with additional query
            for idx, mapping in query_mapping.items():
                existing_hash = hash(' '.join(unique_contexts[idx].split())[:500])
                if existing_hash == content_hash:
                    if "additional_queries" not in mapping:
                        mapping["additional_queries"] = []
                    mapping["additional_queries"].append(query)
                    break
    
    processing_time = time.time() - start_time
    avg_confidence = total_confidence / len(unique_contexts) if unique_contexts else 0.0
    
    return {
        "unique_contexts": unique_contexts,
        "query_mapping": query_mapping,
        "method_used": method,
        "stats": {
            "total_queries": len(queries),
            "total_results": len(results),
            "unique_contents": len(unique_contexts),
            "duplicates_removed": duplicate_count,
            "deduplication_ratio": f"{duplicate_count}/{len(results)}" if results else "0/0",
            "processing_time_ms": round(processing_time * 1000),
            "avg_confidence": round(avg_confidence, 3),
            "method_stats": method_stats
        }
    }

async def process_stream_response(response_stream, session_id: str, client_id: str):
    """Enhanced stream response processing with hybrid retrieval support"""
    chat_history = []
    full_content = ""
    collected_tool_calls = {}
    
    async for chunk in response_stream:
        delta = chunk.choices[0].delta
        
        # Tool calls collection (same as before)
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index
                
                if index not in collected_tool_calls:
                    collected_tool_calls[index] = {
                        'id': '',
                        'type': 'function',
                        'function': {'name': '', 'arguments': ''}
                    }
                
                if tool_call_delta.id:
                    collected_tool_calls[index]['id'] = tool_call_delta.id
                
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        collected_tool_calls[index]['function']['name'] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        collected_tool_calls[index]['function']['arguments'] += tool_call_delta.function.arguments
        
        # Content streaming
        if hasattr(delta, 'content') and delta.content:
            full_content += delta.content
            await manager.send_stream_chunk({
                "type": "content_chunk",
                "content": delta.content
            }, client_id)
    
    # Enhanced tool calls processing
    if collected_tool_calls:
        await manager.send_stream_chunk({
            "type": "tool_calls_start",
            "message": "🔍 Enhanced RAPTOR ile bilgiler aranıyor..."
        }, client_id)
        
        tool_calls_list = []
        for index, tool_call in collected_tool_calls.items():
            tool_calls_list.append({
                'id': tool_call['id'],
                'type': tool_call['type'],
                'function': tool_call['function']
            })
        
        chat_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_list
        })
        
        # Process each tool call with enhanced features
        for tool_call in tool_calls_list:
            function_name = tool_call['function']['name']
            
            try:
                function_args = json.loads(tool_call['function']['arguments'])
            except json.JSONDecodeError:
                print(f"JSON decode error for arguments: {tool_call['function']['arguments']}")
                continue
            
            call_id = tool_call['id']
            
            # 🚀 ENHANCED HYBRID RETRIEVAL
            if function_name == "get_info":
                queries = function_args.get('queries', [])
                
                if not queries:
                    single_query = function_args.get('query', '')
                    if single_query:
                        queries = [single_query]
                
                if not queries:
                    chat_history.append({
                        "role": "tool",
                        "content": json.dumps({"error": "Hiç sorgu belirtilmedi"}),
                        "tool_call_id": call_id
                    })
                    continue
                
                # Show intelligent method selection
                method = "hybrid"  # Default to hybrid for best results
                if len(queries) == 1 and len(queries[0].split()) <= 3:
                    method = "sparse"  # Use sparse for short keyword queries
                elif any(word in q.lower() for q in queries for word in ["define", "what is", "meaning"]):
                    method = "dense"   # Use dense for definitional queries
                
                query_text = ", ".join(queries[:3])
                if len(queries) > 3:
                    query_text += f" ve {len(queries) - 3} tane daha"
                
                await manager.send_stream_chunk({
                    "type": "rag_search",
                    "message": f"🔍 '{query_text}' için {method.upper()} yöntemi ile arama yapıyorum..."
                }, client_id)
                
                try:
                    # 🚀 ENHANCED HYBRID RETRIEVAL
                    retrieve_result = await enhanced_batch_retrieve_async(queries, method)
                    
                    unique_contexts = retrieve_result["unique_contexts"]
                    query_mapping = retrieve_result["query_mapping"]
                    stats = retrieve_result["stats"]
                    method_used = retrieve_result["method_used"]
                    
                    # Enhanced progress message
                    await manager.send_stream_chunk({
                        "type": "rag_complete",
                        "message": f"✅ {method_used.upper()}: {stats['total_queries']} sorgu, {stats['unique_contents']} unique sonuç ({stats['duplicates_removed']} duplikat) - {stats['processing_time_ms']}ms, Güven: {stats['avg_confidence']}"
                    }, client_id)
                    
                    # 🔗 ENHANCED CONTEXT COMBINATION
                    combined_context = f"\n\n--- {method_used.upper()} RETRIEVAL RESULTS ---\n\n"
                    
                    for i, context in enumerate(unique_contexts):
                        mapping = query_mapping.get(i, {})
                        primary_query = mapping.get("primary_query", f"Query {i+1}")
                        additional_queries = mapping.get("additional_queries", [])
                        confidence = mapping.get("confidence", 0.0)
                        hybrid_details = mapping.get("hybrid_details", {})
                        
                        source_info = f"Sorgu: {primary_query}"
                        if additional_queries:
                            source_info += f" (+ {', '.join(additional_queries)})"
                        
                        combined_context += f"=== Result {i+1} ===\n"
                        combined_context += f"Kaynak: {source_info}\n"
                        combined_context += f"Yöntem: {method_used.upper()}\n"
                        combined_context += f"Güven: {confidence:.3f}\n"
                        
                        if hybrid_details:
                            combined_context += f"Hybrid Scores - Dense: {hybrid_details.get('dense_score', 0):.3f}, Sparse: {hybrid_details.get('sparse_score', 0):.3f}, Fused: {hybrid_details.get('fused_score', 0):.3f}\n"
                        
                        combined_context += f"{context}\n\n"
                    
                    chat_history.append({
                        "role": "tool",
                        "content": json.dumps({
                            "queries": queries,
                            "method": method_used,
                            "unique_contexts": unique_contexts,
                            "combined_context": combined_context,
                            "query_mapping": query_mapping,
                            "stats": stats,
                            "hybrid_enabled": True
                        }),
                        "tool_call_id": call_id
                    })
                    
                except Exception as e:
                    chat_history.append({
                        "role": "tool", 
                        "content": json.dumps({
                            "queries": queries,
                            "error": f"Enhanced retrieval sırasında hata: {str(e)}"
                        }),
                        "tool_call_id": call_id
                    })

        # Final response generation (same as before)
        await manager.send_stream_chunk({
            "type": "final_response_start",
            "message": "💭 Enhanced RAPTOR ile cevap hazırlanıyor..."
        }, client_id)
        
        try:
            final_response = await openai_client.chat.completions.create(
                model="gpt-4.1",
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
            
            chat_history.append({
                "role": "assistant",
                "content": final_content
            })
            
            await redis_client.rpush(f"chat_history:{session_id}", json.dumps(chat_history[-1]))
            
        except Exception as e:
            print(f"Final response error: {e}")
            await manager.send_stream_chunk({
                "type": "content_chunk",
                "content": f"Cevap oluştururken hata: {str(e)}"
            }, client_id)
        
        await manager.send_stream_chunk({"type": "stream_end"}, client_id)
        return final_content if 'final_content' in locals() else ""
    
    else:
        # No tool calls, direct content
        if full_content:
            chat_history.append({"role": "assistant", "content": full_content})
            await redis_client.rpush(f"chat_history:{session_id}", json.dumps(chat_history[-1]))
        
        await manager.send_stream_chunk({"type": "stream_end"}, client_id)
        return full_content

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        while True:
            user_message = await websocket.receive_text()
            
            await redis_client.rpush(
                f"chat_history:{session_id}", 
                json.dumps({"role": "user", "content": user_message})
            )
            
            formatted_history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
            formatted_history = [json.loads(msg) for msg in formatted_history]            
            
            await manager.send_stream_chunk({"type": "stream_start"}, client_id)
            
            try:
                response_stream = await openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {
                            "role": "system",
                            "content": """
Sen Enhanced RAPTOR sistemi ile çalışan akıllı bir asistansın. Hybrid retrieval (dense + sparse + query enhancement) 
özelliklerini kullanarak en kaliteli bilgi erişimi sağlayabilirsin.

Özellikler:
- Dense Retrieval: Semantic benzerlik tabanlı arama
- Sparse Retrieval: BM25 keyword tabanlı arama  
- Hybrid Fusion: Her iki yöntemin birleşimi
- Query Enhancement: Sorgu geliştirme ve genişletme
- Reranking: Sonuçların yeniden sıralanması

Kullanıcının sorusuna göre uygun retrieval yöntemini seç ve get_info fonksiyonunu kullan.
                            """
                        }
                    ] + formatted_history,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "get_info",
                                "description": "Enhanced RAPTOR ile bilgi almak için kullanılır. Hybrid retrieval (dense+sparse+reranking) destekler. Karmaşık sorular için birden fazla spesifik arama sorgusu oluştur.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "queries": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Arama sorguları. Kapsamlı konular için birden fazla spesifik sorgu oluştur. Enhanced query processing ve semantic expansion otomatik yapılır.",
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
                
                await process_stream_response(response_stream, session_id, client_id)
                
            except Exception as e:
                print(f"OpenAI API error: {e}")
                await manager.send_stream_chunk({
                    "type": "content_chunk",
                    "content": f"Üzgünüm, bir hata oluştu: {str(e)}"
                }, client_id)
                await manager.send_stream_chunk({"type": "stream_end"}, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(client_id)

@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    history = await redis_client.lrange(f"chat_history:{session_id}", 0, -1)
    return [json.loads(msg) for msg in history]

@app.get("/performance_stats")
async def get_performance_stats():
    """Get Enhanced RAPTOR performance statistics"""
    try:
        stats = RA.get_enhanced_performance_summary()
        return {
            "status": "success",
            "enhanced_raptor_stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test_methods")
async def test_retrieval_methods():
    """Test different retrieval methods"""
    test_queries = [
        "Bu dokümanın ana konusu nedir?",
        "Önemli başlıklar nelerdir?",
        "Ana fikirler"
    ]
    
    results = {}
    for method in ["dense", "sparse", "hybrid"]:
        method_results = []
        for query in test_queries:
            try:
                start_time = time.time()
                context = RA.retrieve_enhanced(query, method=method, top_k=3, max_tokens=1000)
                elapsed = time.time() - start_time
                
                method_results.append({
                    "query": query,
                    "response_time": elapsed,
                    "context_length": len(context),
                    "success": True
                })
            except Exception as e:
                method_results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        results[method] = method_results
    
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Enhanced RAPTOR API with Hybrid Retrieval starting...")
    print("📊 Available endpoints:")
    print("   WebSocket: /ws/{client_id}")
    print("   Performance: /performance_stats")
    print("   Test Methods: /test_methods")
    print("   Chat History: /chat_history/{session_id}")
    uvicorn.run(app, host="0.0.0.0", port=8000)