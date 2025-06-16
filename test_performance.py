# test_performance.py oluşturun
import os
os.environ["OPENAI_API_KEY"] = "None"
# test_performance.py
import asyncio
import time
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.SummarizationModels import GPT41SummarizationModel

async def test_concurrent_performance():
    # AYNI CONFIG'i kullan (build sırasında kullanılan)
    embed_model = CustomEmbeddingModel()
    sum_model = GPT41SummarizationModel()
    
    config = RetrievalAugmentationConfig(
        # Tree ile AYNI embedding model
        embedding_model=embed_model,
        summarization_model=sum_model,
        tree_builder_type="cluster",
        
        # Retrieval optimization
        tr_enable_caching=True,
        tr_adaptive_retrieval=True,
        # tr_similarity_cache_threshold=0.80,  # Optimized
        # tr_cache_ttl=14400,
        tr_top_k=8,
        
        # Performance
        enable_async=True,
        max_concurrent_operations=32,
    )
    
    # Tree'yi config ile load et
    RA = RetrievalAugmentation(config=config, tree="vectordb/raptor-production")
    
    # 10 paralel query test (başlangıç için küçük)
    queries = [f"What is important information about topic {i}?" for i in range(10)]
    
    start = time.time()
    
    # use_async=False ile sync mode
    tasks = [asyncio.to_thread(RA.retrieve, q, collapse_tree=False, use_async=False) for q in queries]
    results = await asyncio.gather(*tasks)
    
    end = time.time()
    
    print(f"10 concurrent queries: {end-start:.2f}s")
    print(f"Average per query: {(end-start)/10:.3f}s")
    print(f"First result preview: {results[0][:200]}...")
    
    # Cache stats
    if hasattr(RA.retriever, 'get_performance_stats'):
        stats = RA.retriever.get_performance_stats()
        print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")

if __name__ == "__main__":
    asyncio.run(test_concurrent_performance())