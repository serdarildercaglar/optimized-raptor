# Enhanced RAPTOR build script with Hybrid Retrieval capabilities
import os
import pathlib
import time
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

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

def create_enhanced_config(document_size: int, use_async: bool = True) -> tuple:
    """
    Create Enhanced RAPTOR configuration with hybrid features
    
    Args:
        document_size: Size of document in characters
        use_async: Whether to use async pipeline
        
    Returns:
        tuple: (ra_config, hybrid_config, parameters_dict)
    """
    
    # Adaptive parameters based on document size
    if document_size < 10000:  # Small documents (< 10KB)
        print("📝 Optimizing for SMALL document...")
        chunk_size = 150
        summarization_length = 200
        num_layers = 3
        batch_size = 50
        
    elif document_size < 100000:  # Medium documents (10KB - 100KB)
        print("📝 Optimizing for MEDIUM document...")
        chunk_size = 120
        summarization_length = 400
        num_layers = 4
        batch_size = 100
        
    else:  # Large documents (> 100KB)
        print("📝 Optimizing for LARGE document...")
        chunk_size = 100
        summarization_length = 512
        num_layers = 5
        batch_size = 150
    
    from raptor import RetrievalAugmentationConfig, GPT4OSummarizationModel
    from raptor.EmbeddingModels import CustomEmbeddingModel
    from raptor.enhanced_retrieval_augmentation import HybridConfig
    from raptor.hybrid_retriever import FusionMethod
    
    # Initialize optimized models
    embed_model = CustomEmbeddingModel()
    sum_model = GPT4OSummarizationModel()
    
    # Standard RAPTOR configuration
    ra_config = RetrievalAugmentationConfig(
        # ===== ENHANCED TEXT CHUNKING PARAMETERS =====
        tb_max_tokens=chunk_size,
        tb_summarization_length=summarization_length,
        tb_num_layers=num_layers,
        
        # ===== QUALITY-FOCUSED CLUSTERING PARAMETERS =====
        tb_threshold=0.35,
        tb_top_k=7,
        tb_selection_mode="top_k",
        
        # ===== ASYNC PERFORMANCE PARAMETERS =====
        tb_batch_size=batch_size,
        tb_build_mode="async" if use_async else "sync",
        tb_enable_progress_tracking=True,
        
        # ===== RETRIEVAL OPTIMIZATION PARAMETERS =====
        tr_threshold=0.4,
        tr_top_k=8,
        tr_selection_mode="top_k",
        tr_enable_caching=True,
        tr_adaptive_retrieval=True,
        tr_early_termination=True,
        
        # ===== PIPELINE PERFORMANCE SETTINGS =====
        enable_async=use_async,
        enable_caching=True,
        enable_metrics=True,
        enable_progress_tracking=True,
        performance_monitoring=True,
        max_concurrent_operations=12,
        cache_ttl=7200,
        
        # ===== MODEL CONFIGURATION =====
        summarization_model=sum_model,
        embedding_model=embed_model,
        
        # ===== TREE BUILDER TYPE =====
        tree_builder_type="cluster",
    )
    
    # Enhanced Hybrid Configuration
    hybrid_config = HybridConfig(
        # ===== HYBRID RETRIEVAL FEATURES =====
        enable_hybrid=True,
        enable_query_enhancement=True,
        enable_sparse_retrieval=True,
        enable_reranking=True,
        
        # ===== FUSION SETTINGS =====
        fusion_method=FusionMethod.RRF,  # Reciprocal Rank Fusion for best results
        dense_weight=0.6,  # Favor dense retrieval slightly
        sparse_weight=0.4,
        
        # ===== SPARSE RETRIEVAL SETTINGS =====
        sparse_algorithm="bm25_okapi",  # Most effective BM25 variant
        sparse_k1=1.2,  # BM25 term frequency saturation
        sparse_b=0.75,  # BM25 length normalization
        
        # ===== QUERY ENHANCEMENT SETTINGS =====
        max_query_expansions=5,
        semantic_expansion=True,
        
        # ===== RERANKING SETTINGS =====
        rerank_top_k=20,  # Rerank top 20 results
        
        # ===== PERFORMANCE SETTINGS =====
        enable_caching=True,
        cache_dir="hybrid_cache"
    )
    
    # Return both configs and parameters for logging
    parameters = {
        'chunk_size': chunk_size,
        'summarization_length': summarization_length,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'document_type': 'small' if document_size < 10000 else 'medium' if document_size < 100000 else 'large',
        'hybrid_features': {
            'fusion_method': 'RRF',
            'sparse_algorithm': 'BM25-Okapi',
            'query_enhancement': True,
            'reranking': True
        }
    }
    
    return ra_config, hybrid_config, parameters

def build_and_evaluate_enhanced_raptor(
    text: str, 
    save_path: str = "vectordb/enhanced-raptor-optimized",
    run_evaluation: bool = True
) -> 'EnhancedRetrievalAugmentation':
    """
    Build Enhanced RAPTOR with hybrid capabilities and run evaluation
    
    Args:
        text: Input text to process
        save_path: Path to save the tree
        run_evaluation: Whether to run performance evaluation
    
    Returns:
        EnhancedRetrievalAugmentation instance
    """
    
    print("🚀 Building Enhanced RAPTOR with HYBRID RETRIEVAL capabilities...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create enhanced configuration
    ra_config, hybrid_config, params = create_enhanced_config(len(text), use_async=True)
    
    from raptor.enhanced_retrieval_augmentation import EnhancedRetrievalAugmentation
    
    # Initialize Enhanced RAPTOR
    enhanced_RA = EnhancedRetrievalAugmentation(
        config=ra_config,
        hybrid_config=hybrid_config
    )
    enhanced_RA.set_progress_callback(progress_callback)
    
    print(f"🏗️ Processing {len(text):,} characters...")
    print(f"⚙️ Configuration: {params['chunk_size']} tokens/chunk, "
          f"{params['num_layers']} layers, "
          f"batch_size={params['batch_size']}, "
          f"type={params['document_type']}")
    print(f"🔄 Hybrid Features: {params['hybrid_features']}")
    print("-" * 80)
    
    # Build tree with Enhanced RAPTOR
    try:
        enhanced_RA.add_documents(text)
        build_time = time.time() - start_time
        
        print("-" * 80)
        print(f"✅ Enhanced tree construction completed in {build_time:.1f}s!")
        
        # Performance evaluation with hybrid features
        if run_evaluation:
            print("\n🔍 Running enhanced performance evaluation...")
            evaluation_results = run_enhanced_performance_evaluation(enhanced_RA)
            
            # Print results
            print("\n📊 ENHANCED PERFORMANCE RESULTS:")
            print("=" * 50)
            for metric, value in evaluation_results.items():
                print(f"   {metric}: {value}")
        
        # Save enhanced tree
        print(f"\n💾 Saving enhanced tree to {save_path}...")
        ensure_directory_exists(save_path)
        enhanced_RA.save(save_path, include_metadata=True)
        print(f"✅ Enhanced tree saved successfully!")
        
        # Enhanced summary
        perf_summary = enhanced_RA.get_enhanced_performance_summary()
        print("\n🎯 ENHANCED FINAL SUMMARY:")
        print("=" * 50)
        
        # Tree statistics
        if 'tree_stats' in perf_summary:
            stats = perf_summary['tree_stats']
            print(f"   Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"   Layers Built: {stats.get('num_layers', 0)}")
            print(f"   Leaf Nodes: {stats.get('leaf_nodes', 0)}")
            print(f"   Root Nodes: {stats.get('root_nodes', 0)}")
        
        # Hybrid features status
        if 'hybrid_features' in perf_summary:
            features = perf_summary['hybrid_features']
            print(f"   🔄 Hybrid Features:")
            for feature, enabled in features.items():
                status = "✅" if enabled else "❌"
                print(f"      {status} {feature.replace('_', ' ').title()}")
        
        print(f"   Build Time: {build_time:.1f}s")
        print(f"   🚀 Enhanced RAPTOR ready for production! 🚀")
        
        return enhanced_RA
        
    except Exception as e:
        print(f"❌ Error during enhanced tree construction: {e}")
        raise

def run_enhanced_performance_evaluation(enhanced_RA) -> dict:
    """Run comprehensive performance evaluation with hybrid features"""
    
    results = {}
    
    # Test queries for evaluation
    test_queries = [
        "Bu dokümanın ana konusu nedir?",
        "What are the main topics discussed?", 
        "Summarize the key points",
        "What is the most important information?",
        "Ana başlıklar nelerdir?",
    ]
    
    # 1. Enhanced Retrieval Performance Test (Multiple Methods)
    print("   Testing enhanced retrieval performance (dense, sparse, hybrid)...")
    
    methods_performance = {}
    for method in ["dense", "sparse", "hybrid"]:
        method_times = []
        context_lengths = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                context = enhanced_RA.retrieve_enhanced(
                    query, 
                    method=method, 
                    top_k=5, 
                    max_tokens=2000,
                    enhance_query=True
                )
                retrieval_time = time.time() - start_time
                
                method_times.append(retrieval_time)
                context_lengths.append(len(context))
                
            except Exception as e:
                print(f"      Warning: {method} method failed for query '{query}': {e}")
                continue
        
        if method_times:
            methods_performance[method] = {
                'avg_time': sum(method_times) / len(method_times),
                'avg_context_length': sum(context_lengths) / len(context_lengths),
                'success_rate': len(method_times) / len(test_queries)
            }
    
    # Find best performing method
    if methods_performance:
        best_method = min(methods_performance.keys(), 
                         key=lambda m: methods_performance[m]['avg_time'])
        results['Best Method'] = f"{best_method} ({methods_performance[best_method]['avg_time']:.3f}s)"
        
        for method, perf in methods_performance.items():
            results[f'{method.title()} Avg Time'] = f"{perf['avg_time']:.3f}s"
            results[f'{method.title()} Success Rate'] = f"{perf['success_rate']:.1%}"
    
    # 2. Query Enhancement Test
    print("   Testing query enhancement capabilities...")
    if hasattr(enhanced_RA, 'query_enhancer') and enhanced_RA.query_enhancer:
        try:
            enhanced_query = enhanced_RA.enhance_query_only("machine learning")
            results['Query Enhancement'] = f"✅ {enhanced_query.intent.value} intent, {len(enhanced_query.expanded_terms)} expansions"
        except Exception as e:
            results['Query Enhancement'] = f"❌ Failed: {str(e)}"
    else:
        results['Query Enhancement'] = "❌ Not initialized"
    
    # 3. Hybrid Cache Performance Test
    print("   Testing enhanced cache performance...")
    cache_start = time.time()
    try:
        cached_result = enhanced_RA.retrieve_enhanced(
            test_queries[0], 
            method="hybrid", 
            top_k=5, 
            max_tokens=2000
        )
        cache_time = time.time() - cache_start
        results['Hybrid Cache Time'] = f"{cache_time:.3f}s"
    except Exception as e:
        results['Hybrid Cache Time'] = f"Failed: {str(e)}"
    
    # 4. Method Comparison Test
    print("   Testing method comparison capabilities...")
    try:
        comparison = enhanced_RA.compare_retrieval_methods(
            "What is the main topic?", 
            top_k=3
        )
        successful_methods = len([m for m, r in comparison['results'].items() if 'error' not in r])
        results['Method Comparison'] = f"✅ {successful_methods}/3 methods successful"
    except Exception as e:
        results['Method Comparison'] = f"❌ Failed: {str(e)}"
    
    # 5. Enhanced QA Performance Test
    print("   Testing enhanced QA performance...")
    qa_start = time.time()
    try:
        answer = enhanced_RA.answer_question(
            "Bu dokümanın en önemli bilgisi nedir?",
            use_async=True,
            top_k=5,
            max_tokens=2000
        )
        qa_time = time.time() - qa_start
        results['Enhanced QA Time'] = f"{qa_time:.3f}s"
        results['Enhanced Answer Length'] = f"{len(answer)} chars"
    except Exception as e:
        results['Enhanced QA Time'] = f"Failed: {str(e)}"
    
    # 6. Hybrid Statistics
    perf_summary = enhanced_RA.get_enhanced_performance_summary()
    if 'hybrid_retriever' in perf_summary:
        hybrid_stats = perf_summary['hybrid_retriever']
        results['Hybrid Retrievals'] = hybrid_stats.get('total_retrievals', 0)
        results['Fusion Method'] = hybrid_stats.get('fusion_method', 'N/A')
    
    return results

def ensure_directory_exists(file_path: str):
    """Ensure directory exists for file path"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory: {directory}")

def main():
    """Main execution function with enhanced features"""
    
    print("🎯 Enhanced RAPTOR Build Script with HYBRID RETRIEVAL")
    print("Features: Dense + Sparse + Query Enhancement + Reranking + Fusion")
    print("=" * 80)
    
    try:
        # Load data
        text = load_text_data('data.txt')
        
        # Build and evaluate Enhanced RAPTOR
        enhanced_RA = build_and_evaluate_enhanced_raptor(
            text=text,
            save_path="vectordb/enhanced-raptor-optimized",
            run_evaluation=True
        )
        
        print("\n🎉 ENHANCED RAPTOR BUILD COMPLETED SUCCESSFULLY!")
        print("Your Enhanced RAPTOR with Hybrid Retrieval is ready for production use.")
        
        # Optional: Quick hybrid feature test
        print("\n🧪 Quick hybrid feature test...")
        try:
            # Test different methods
            test_query = "Bu dokümanın özeti nedir?"
            
            for method in ["dense", "sparse", "hybrid"]:
                start_time = time.time()
                test_result = enhanced_RA.retrieve_enhanced(
                    test_query,
                    method=method,
                    top_k=3,
                    max_tokens=500
                )
                elapsed = time.time() - start_time
                print(f"✅ {method.upper()} Test: {elapsed:.3f}s - {len(test_result)} chars")
        
        except Exception as e:
            print(f"⚠️ Hybrid test encountered issue: {e}")
        
        # Performance summary
        perf_summary = enhanced_RA.get_enhanced_performance_summary()
        if 'hybrid_features' in perf_summary:
            print(f"\n🔄 Hybrid Features Status:")
            for feature, enabled in perf_summary['hybrid_features'].items():
                status = "✅" if enabled else "❌"
                print(f"   {status} {feature.replace('_', ' ').title()}")
        
    except Exception as e:
        print(f"\n❌ Enhanced build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Enhanced RAPTOR with Hybrid Retrieval is now ready!")
        print("📝 Use test.py to interact with the enhanced system.")
    else:
        print("\n❌ Build failed. Please check the error messages above.")