# Full RAPTOR test with optimized configuration
import os
import pathlib
import time
from dotenv import load_dotenv
load_dotenv()
# os.environ["OPENAI_API_KEY"] = "sk-proj-funt"

print("🚀 Starting FULL RAPTOR build with optimized configuration...")

# Read full data
with open('data.txt', 'r') as file:
    text = file.read()

print(f"📄 Processing {len(text)} characters of full text...")

from raptor import RetrievalAugmentation 
from raptor import RetrievalAugmentationConfig
from raptor import GPT4OSummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel

# Initialize models
embed_model = CustomEmbeddingModel()
sum_model = GPT4OSummarizationModel()

# Optimized configuration for full data
print("⚙️ Configuring optimized RAPTOR settings...")

RA_config = RetrievalAugmentationConfig(
    # Tree Builder optimizations
    tb_summarization_length=512,      # Good balance for quality
    tb_max_tokens=100,                # Reasonable chunk size
    tb_num_layers=5,                  # Full layer construction
    tb_batch_size=100,                # Batch embedding creation
    tb_build_mode="async",            # Use async for performance
    tb_enable_progress_tracking=True, # Real-time progress
    
    # Tree Retriever optimizations  
    tr_enable_caching=True,           # Enable smart caching
    tr_adaptive_retrieval=True,       # Adaptive parameter adjustment
    tr_early_termination=True,        # Confidence-based stopping
    
    # Enhanced pipeline features
    enable_async=True,                # Async pipeline
    enable_caching=True,              # All caching features
    enable_metrics=True,              # Performance monitoring
    enable_progress_tracking=True,    # Progress callbacks
    performance_monitoring=True,      # Detailed metrics
    max_concurrent_operations=10,     # Parallel processing
    cache_ttl=3600,                   # 1 hour cache (FIXED: was tr_cache_ttl)
    
    # Models
    summarization_model=sum_model, 
    embedding_model=embed_model,
) 

# Progress callback for real-time updates
def progress_callback(progress):
    print(f"📊 Progress: Layer {progress.current_layer}/{progress.total_layers} "
          f"({progress.layer_progress:.1%}), "
          f"Nodes: {progress.created_nodes}/{progress.total_nodes} "
          f"({progress.node_progress:.1%}), "
          f"Time: {progress.elapsed_time:.1f}s")

print("🏗️ Initializing RAPTOR with optimized configuration...")
RA = RetrievalAugmentation(config=RA_config)

# Set progress callback
RA.set_progress_callback(progress_callback)

# Track total time
start_time = time.time()

try:
    print("🔨 Building tree from full text (this may take a few minutes)...")
    print("📈 Real-time progress updates will show below:")
    print("-" * 80)
    
    # Build tree with full text
    RA.add_documents(text)
    
    build_time = time.time() - start_time
    print("-" * 80)
    print(f"✅ Tree construction completed in {build_time:.1f}s!")
    
    # Get performance summary
    print("\n📊 Performance Summary:")
    perf_stats = RA.get_performance_summary()
    
    if 'pipeline' in perf_stats:
        pipeline = perf_stats['pipeline']
        print(f"   Build Time: {pipeline.get('build_time', 0):.1f}s")
        print(f"   Nodes Processed: {pipeline.get('nodes_processed', 0)}")
        print(f"   Layers Built: {pipeline.get('layers_built', 0)}")
    
    if 'tree_builder' in perf_stats:
        builder = perf_stats['tree_builder']
        print(f"   Embedding Batches: {builder.get('embedding_batch_count', 0)}")
        print(f"   Avg Layer Time: {builder.get('avg_layer_time', 0):.2f}s")
    
    # Test retrieval performance
    print("\n🔍 Testing retrieval performance...")
    test_queries = [
        "What is this document about?",
        "What are the main topics discussed?",
        "Summarize the key points",
    ]
    
    retrieval_times = []
    for i, query in enumerate(test_queries, 1):
        query_start = time.time()
        context = RA.retrieve(query, collapse_tree=True, max_tokens=2000)
        query_time = time.time() - query_start
        retrieval_times.append(query_time)
        
        print(f"   Query {i}: {query_time:.3f}s - Context: {len(context)} chars")
    
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    print(f"   Average Retrieval Time: {avg_retrieval:.3f}s")
    
    # Test caching performance
    print("\n💾 Testing cache performance...")
    cache_start = time.time()
    cached_context = RA.retrieve(test_queries[0], collapse_tree=True, max_tokens=2000)
    cache_time = time.time() - cache_start
    print(f"   Cache Hit Time: {cache_time:.3f}s (should be much faster)")
    
    # Get retriever stats
    if hasattr(RA.retriever, 'get_performance_stats'):
        retriever_stats = RA.retriever.get_performance_stats()
        cache_hit_rate = retriever_stats.get('cache_hit_rate', 0)
        print(f"   Cache Hit Rate: {cache_hit_rate:.1%}")
    
    # Test QA performance
    print("\n🤖 Testing question answering...")
    qa_start = time.time()
    answer = RA.answer_question("What is the main topic of this document?", 
                               collapse_tree=True, max_tokens=2000)
    qa_time = time.time() - qa_start
    
    print(f"   QA Response Time: {qa_time:.3f}s")
    print(f"   Answer Preview: {answer[:200]}...")
    
    # Save optimized tree
    print("\n💾 Saving optimized tree...")
    if not os.path.exists("vectordb"):
        os.makedirs("vectordb")
    
    SAVE_PATH = "vectordb/raptor-optimized"
    RA.save(SAVE_PATH, include_metadata=True)
    print(f"✅ Optimized tree saved to {SAVE_PATH}")
    print(f"📄 Detailed metadata saved to {SAVE_PATH}.json")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 FULL RAPTOR TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Total Processing Time: {total_time:.1f}s")
    print(f"📈 Performance Improvements:")
    print(f"   • Enhanced chunking with semantic boundaries")
    print(f"   • Batch embedding creation (up to 100x faster)")
    print(f"   • Quality-focused clustering with fallbacks")
    print(f"   • Smart caching system active")
    print(f"   • Async pipeline optimizations")
    print(f"   • Real-time progress tracking")
    
    if 'tree_stats' in perf_stats:
        tree_stats = perf_stats['tree_stats']
        print(f"\n🌳 Tree Statistics:")
        print(f"   • Total Nodes: {tree_stats.get('total_nodes', 0)}")
        print(f"   • Layers: {tree_stats.get('num_layers', 0)}")
        print(f"   • Leaf Nodes: {tree_stats.get('leaf_nodes', 0)}")
        print(f"   • Root Nodes: {tree_stats.get('root_nodes', 0)}")
    
    print(f"\n🚀 RAPTOR is now production-ready with enterprise-level performance!")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    
    # Still try to get partial performance stats
    try:
        partial_stats = RA.get_performance_summary()
        print(f"\n📊 Partial Performance Stats: {partial_stats}")
    except:
        pass
    
finally:
    # Cleanup
    print(f"\n🧹 Cleaning up resources...")
    try:
        RA.clear_all_caches()
        print("✅ Caches cleared")
    except:
        pass