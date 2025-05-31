# Optimized RAPTOR build script with enhanced text chunking parameters
import os
import pathlib
import time
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
# OpenAI API key - replace with your actual key
# os.environ["OPENAI_API_KEY"] = "sk-proj-funtMJyTm_f"

def load_text_data(file_path: str = 'data.txt') -> str:
    """Load text data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"📄 Loaded {len(text):,} characters from {file_path}")
        return text
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        raise

def progress_callback(progress):
    """Enhanced progress callback with detailed information"""
    print(f"📊 Layer {progress.current_layer}/{progress.total_layers} "
          f"({progress.layer_progress:.1%}) | "
          f"Nodes: {progress.created_nodes}/{progress.total_nodes} "
          f"({progress.node_progress:.1%}) | "
          f"Embedding: {progress.embedding_time:.1f}s | "
          f"Total: {progress.elapsed_time:.1f}s")

def create_optimized_config(document_size: int, use_async: bool = True) -> tuple:
    """
    Create optimized configuration based on document characteristics
    
    Args:
        document_size: Size of document in characters
        use_async: Whether to use async pipeline
        
    Returns:
        tuple: (config, parameters_dict) where parameters_dict contains the settings used
    """
    
    # Adaptive parameters based on document size
    if document_size < 10000:  # Small documents (< 10KB)
        print("📝 Optimizing for SMALL document...")
        chunk_size = 150        # Larger chunks for better context
        summarization_length = 200
        num_layers = 3
        batch_size = 50
        
    elif document_size < 100000:  # Medium documents (10KB - 100KB)
        print("📝 Optimizing for MEDIUM document...")
        chunk_size = 120       # Balanced chunk size
        summarization_length = 400
        num_layers = 4
        batch_size = 100
        
    else:  # Large documents (> 100KB)
        print("📝 Optimizing for LARGE document...")
        chunk_size = 100       # Smaller chunks for better granularity
        summarization_length = 512
        num_layers = 5
        batch_size = 150
    
    from raptor import RetrievalAugmentationConfig
    from raptor import GPT4OSummarizationModel
    from raptor.EmbeddingModels import CustomEmbeddingModel
    
    # Initialize optimized models
    embed_model = CustomEmbeddingModel()
    sum_model = GPT4OSummarizationModel()
    
    config = RetrievalAugmentationConfig(
        # ===== ENHANCED TEXT CHUNKING PARAMETERS =====
        tb_max_tokens=chunk_size,              # Optimized for semantic chunking
        tb_summarization_length=summarization_length,
        tb_num_layers=num_layers,
        
        # ===== QUALITY-FOCUSED CLUSTERING PARAMETERS =====
        tb_threshold=0.35,                     # Lowered for better clustering sensitivity
        tb_top_k=7,                           # Increased for richer context
        tb_selection_mode="top_k",            # More predictable than threshold mode
        
        # ===== ASYNC PERFORMANCE PARAMETERS =====
        tb_batch_size=batch_size,             # Optimized batch processing
        tb_build_mode="async" if use_async else "sync",
        tb_enable_progress_tracking=True,
        
        # ===== RETRIEVAL OPTIMIZATION PARAMETERS =====
        tr_threshold=0.4,                     # Balanced retrieval threshold
        tr_top_k=8,                          # More diverse retrieval
        tr_selection_mode="top_k",
        tr_enable_caching=True,
        tr_adaptive_retrieval=True,          # Enable adaptive parameter adjustment
        tr_early_termination=True,           # Smart stopping
        
        # ===== PIPELINE PERFORMANCE SETTINGS =====
        enable_async=use_async,
        enable_caching=True,
        enable_metrics=True,
        enable_progress_tracking=True,
        performance_monitoring=True,
        max_concurrent_operations=12,        # Increased concurrency
        cache_ttl=7200,                     # 2 hour cache for better persistence
        
        # ===== MODEL CONFIGURATION =====
        summarization_model=sum_model,
        embedding_model=embed_model,
        
        # ===== TREE BUILDER TYPE =====
        tree_builder_type="cluster",        # Use enhanced cluster builder
    )
    
    # Return both config and parameters for logging
    parameters = {
        'chunk_size': chunk_size,
        'summarization_length': summarization_length,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'document_type': 'small' if document_size < 10000 else 'medium' if document_size < 100000 else 'large'
    }
    
    return config, parameters

def build_and_evaluate_raptor(
    text: str, 
    save_path: str = "vectordb/raptor-optimized",
    run_evaluation: bool = True
) -> 'RetrievalAugmentation':
    """
    Build RAPTOR tree with optimized parameters and run evaluation
    
    Args:
        text: Input text to process
        save_path: Path to save the tree
        run_evaluation: Whether to run performance evaluation
    
    Returns:
        RetrievalAugmentation instance
    """
    
    print("🚀 Building RAPTOR with OPTIMIZED parameters...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create optimized configuration and get parameters
    config, params = create_optimized_config(len(text), use_async=True)
    
    from raptor import RetrievalAugmentation
    
    # Initialize RAPTOR
    RA = RetrievalAugmentation(config=config)
    RA.set_progress_callback(progress_callback)
    
    print(f"🏗️ Processing {len(text):,} characters...")
    print(f"⚙️ Configuration: {params['chunk_size']} tokens/chunk, "
          f"{params['num_layers']} layers, "
          f"batch_size={params['batch_size']}, "
          f"type={params['document_type']}")
    print("-" * 80)
    
    # Build tree
    try:
        RA.add_documents(text)
        build_time = time.time() - start_time
        
        print("-" * 80)
        print(f"✅ Tree construction completed in {build_time:.1f}s!")
        
        # Performance evaluation
        if run_evaluation:
            print("\n🔍 Running performance evaluation...")
            evaluation_results = run_performance_evaluation(RA)
            
            # Print results
            print("\n📊 PERFORMANCE RESULTS:")
            print("=" * 50)
            for metric, value in evaluation_results.items():
                print(f"   {metric}: {value}")
        
        # Save tree
        print(f"\n💾 Saving optimized tree to {save_path}...")
        ensure_directory_exists(save_path)
        RA.save(save_path, include_metadata=True)
        print(f"✅ Tree saved successfully!")
        
        # Summary
        perf_summary = RA.get_performance_summary()
        print("\n🎯 FINAL SUMMARY:")
        print("=" * 50)
        if 'tree_stats' in perf_summary:
            stats = perf_summary['tree_stats']
            print(f"   Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"   Layers Built: {stats.get('num_layers', 0)}")
            print(f"   Leaf Nodes: {stats.get('leaf_nodes', 0)}")
            print(f"   Root Nodes: {stats.get('root_nodes', 0)}")
        
        print(f"   Build Time: {build_time:.1f}s")
        print(f"   Ready for production use! 🚀")
        
        return RA
        
    except Exception as e:
        print(f"❌ Error during tree construction: {e}")
        raise

def run_performance_evaluation(RA) -> dict:
    """Run comprehensive performance evaluation"""
    
    results = {}
    
    # Test queries for evaluation
    test_queries = [
        "Bu dokümanın ana konusu nedir?",
        "What are the main topics discussed?",
        "Summarize the key points",
        "What is the most important information?",
        "Ana başlıklar nelerdir?",
    ]
    
    # 1. Retrieval Performance Test
    print("   Testing retrieval performance...")
    retrieval_times = []
    context_lengths = []
    
    for query in test_queries:
        start_time = time.time()
        context = RA.retrieve(query, collapse_tree=True, max_tokens=3000)
        retrieval_time = time.time() - start_time
        
        retrieval_times.append(retrieval_time)
        context_lengths.append(len(context))
    
    results['Avg Retrieval Time'] = f"{sum(retrieval_times) / len(retrieval_times):.3f}s"
    results['Avg Context Length'] = f"{sum(context_lengths) / len(context_lengths):.0f} chars"
    
    # 2. Cache Performance Test
    print("   Testing cache performance...")
    cache_start = time.time()
    cached_result = RA.retrieve(test_queries[0], collapse_tree=True, max_tokens=3000)
    cache_time = time.time() - cache_start
    
    results['Cache Hit Time'] = f"{cache_time:.3f}s"
    
    # Get cache statistics
    if hasattr(RA.retriever, 'get_performance_stats'):
        cache_stats = RA.retriever.get_performance_stats()
        results['Cache Hit Rate'] = f"{cache_stats.get('cache_hit_rate', 0):.1%}"
    
    # 3. QA Performance Test  
    print("   Testing QA performance...")
    qa_start = time.time()
    answer = RA.answer_question(
        "Bu dokümanın en önemli bilgisi nedir?",
        collapse_tree=True,
        max_tokens=2000
    )
    qa_time = time.time() - qa_start
    
    results['QA Response Time'] = f"{qa_time:.3f}s"
    results['Answer Length'] = f"{len(answer)} chars"
    
    # 4. Tree Statistics
    perf_summary = RA.get_performance_summary()
    if 'tree_builder' in perf_summary:
        builder_stats = perf_summary['tree_builder']
        results['Embedding Efficiency'] = f"{builder_stats.get('embedding_efficiency', 0):.1f} batches/sec"
        results['Nodes per Second'] = f"{builder_stats.get('nodes_per_second', 0):.1f}"
    
    return results

def ensure_directory_exists(file_path: str):
    """Ensure directory exists for file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory: {directory}")

def main():
    """Main execution function"""
    
    print("🎯 RAPTOR Optimized Build Script")
    print("Enhanced for semantic chunking and quality clustering")
    print("=" * 80)
    
    try:
        # Load data
        text = load_text_data('data.txt')
        
        # Build and evaluate
        RA = build_and_evaluate_raptor(
            text=text,
            save_path="vectordb/raptor-optimized",
            run_evaluation=True
        )
        
        print("\n🎉 RAPTOR BUILD COMPLETED SUCCESSFULLY!")
        print("Your optimized RAPTOR tree is ready for production use.")
        
        # Optional: Quick test
        print("\n🧪 Quick functionality test...")
        test_answer = RA.answer_question(
            "Bu dokümanın özeti nedir?",
            max_tokens=1500
        )
        print(f"✅ QA Test successful - Answer: {test_answer[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()