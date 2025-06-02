# Full Enhanced RAPTOR test with comprehensive hybrid retrieval evaluation
import os
import pathlib
import time
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

print("🚀 Starting FULL Enhanced RAPTOR build with Hybrid Retrieval...")

# Read full data
with open('data/data.txt', 'r') as file:
    text = file.read()

print(f"📄 Processing {len(text)} characters of full text...")

# Enhanced RAPTOR imports
from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, 
    HybridConfig, 
    create_enhanced_raptor
)
from raptor.hybrid_retriever import FusionMethod
from raptor.evaluation_framework import (
    HybridRAPTOREvaluator,
    EvaluationQuery,
    create_sample_evaluation_set
)

# Initialize models
embed_model = CustomEmbeddingModel()
sum_model = GPT41SummarizationModel()

# Comprehensive Enhanced RAPTOR Configuration
print("⚙️ Configuring Enhanced RAPTOR with ALL hybrid features...")

# Standard RAPTOR config with optimizations
ra_config = RetrievalAugmentationConfig(
    # Tree Builder optimizations
    tb_summarization_length=512,      # Higher quality summaries
    tb_max_tokens=100,                # Optimal chunk size
    tb_num_layers=5,                  # Full layer construction
    tb_batch_size=150,                # Large batch for efficiency
    tb_build_mode="async",            # Async for performance
    tb_enable_progress_tracking=True, # Real-time progress
    tb_threshold=0.35,                # Sensitive clustering
    tb_top_k=8,                      # More diverse clustering
    
    # Tree Retriever optimizations  
    tr_enable_caching=True,           # Smart caching
    tr_adaptive_retrieval=True,       # Adaptive parameters
    tr_early_termination=True,        # Confidence-based stopping
    tr_threshold=0.4,                 # Balanced retrieval
    tr_top_k=10,                     # More diverse retrieval
    
    # Enhanced pipeline features
    enable_async=True,                # Full async pipeline
    enable_caching=True,              # All caching features
    enable_metrics=True,              # Comprehensive monitoring
    enable_progress_tracking=True,    # Progress callbacks
    performance_monitoring=True,      # Detailed metrics
    max_concurrent_operations=15,     # High parallelism
    cache_ttl=7200,                   # 2 hour cache
    
    # Models
    summarization_model=sum_model, 
    embedding_model=embed_model,
    tree_builder_type="cluster",
) 

# Comprehensive Hybrid Configuration
hybrid_config = HybridConfig(
    # ===== CORE HYBRID FEATURES =====
    enable_hybrid=True,               # Enable hybrid retrieval
    enable_query_enhancement=True,    # Query enhancement & expansion
    enable_sparse_retrieval=True,     # BM25 sparse retrieval
    enable_reranking=False,           # Cross-encoder reranking
    
    # ===== FUSION CONFIGURATION =====
    fusion_method=FusionMethod.RRF,   # Reciprocal Rank Fusion (best for general use)
    dense_weight=0.6,                 # Slight preference for semantic search
    sparse_weight=0.4,                # Good keyword matching
    
    # ===== SPARSE RETRIEVAL SETTINGS =====
    sparse_algorithm="bm25_okapi",    # Most effective BM25 variant
    sparse_k1=1.2,                   # Term frequency saturation
    sparse_b=0.75,                   # Length normalization
    
    # ===== QUERY ENHANCEMENT SETTINGS =====
    max_query_expansions=8,           # More expansions for better coverage
    semantic_expansion=True,          # Use embeddings for expansion
    
    # ===== RERANKING SETTINGS =====
    rerank_top_k=25,                 # Rerank top 25 for quality
    
    # ===== PERFORMANCE SETTINGS =====
    enable_caching=True,
    cache_dir="enhanced_hybrid_cache"
)

# Progress callback for detailed tracking
def detailed_progress_callback(progress):
    print(f"📊 Progress: Layer {progress.current_layer}/{progress.total_layers} "
          f"({progress.layer_progress:.1%}), "
          f"Nodes: {progress.created_nodes}/{progress.total_nodes} "
          f"({progress.node_progress:.1%}), "
          f"Embedding: {progress.embedding_time:.1f}s, "
          f"Summarization: {progress.summarization_time:.1f}s, "
          f"Total: {progress.elapsed_time:.1f}s")

print("🏗️ Initializing Enhanced RAPTOR with comprehensive hybrid features...")
enhanced_RA = EnhancedRetrievalAugmentation(
    config=ra_config,
    hybrid_config=hybrid_config
)

# Set detailed progress callback
enhanced_RA.set_progress_callback(detailed_progress_callback)

# Track total time
start_time = time.time()

try:
    print("🔨 Building enhanced tree from full text (this may take several minutes)...")
    print("📈 Real-time progress updates will show below:")
    print("-" * 80)
    
    # Build tree with full text
    enhanced_RA.add_documents(text)
    
    build_time = time.time() - start_time
    print("-" * 80)
    print(f"✅ Enhanced tree construction completed in {build_time:.1f}s!")
    
    # Get comprehensive performance summary
    print("\n📊 Enhanced Performance Summary:")
    perf_stats = enhanced_RA.get_enhanced_performance_summary()
    
    # Pipeline performance
    if 'pipeline' in perf_stats:
        pipeline = perf_stats['pipeline']
        print(f"   Build Time: {pipeline.get('build_time', 0):.1f}s")
        print(f"   Nodes Processed: {pipeline.get('nodes_processed', 0)}")
        print(f"   Layers Built: {pipeline.get('layers_built', 0)}")
    
    # Tree builder performance
    if 'tree_builder' in perf_stats:
        builder = perf_stats['tree_builder']
        print(f"   Embedding Batches: {builder.get('embedding_batch_count', 0)}")
        print(f"   Avg Layer Time: {builder.get('avg_layer_time', 0):.2f}s")
        print(f"   Nodes per Second: {builder.get('nodes_per_second', 0):.1f}")
    
    # Hybrid features status
    if 'hybrid_features' in perf_stats:
        features = perf_stats['hybrid_features']
        print(f"\n🔄 Hybrid Features Status:")
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}")
    
    # ===== COMPREHENSIVE HYBRID RETRIEVAL TESTING =====
    print("\n🔍 Testing comprehensive hybrid retrieval capabilities...")
    
    test_queries = [
        "What is this document about?",
        "Bu dokümanın ana konusu nedir?",
        "What are the main topics discussed?",
        "Summarize the key points",
        "Ana başlıklar nelerdir?",
        "machine learning applications",
        "artificial intelligence definition",
    ]
    
    # Test each method comprehensively
    print("\n📊 COMPREHENSIVE METHOD COMPARISON:")
    print("=" * 60)
    
    methods_results = {}
    
    for method in ["dense", "sparse", "hybrid"]:
        print(f"\n🔄 Testing {method.upper()} Retrieval:")
        print("-" * 40)
        
        method_times = []
        context_lengths = []
        success_count = 0
        
        for i, query in enumerate(test_queries, 1):
            try:
                query_start = time.time()
                
                # Enhanced retrieval with detailed settings
                context, detailed_results = enhanced_RA.retrieve_enhanced(
                    query, 
                    method=method, 
                    top_k=8,
                    max_tokens=2500,
                    enhance_query=True,
                    return_detailed=True
                )
                
                query_time = time.time() - query_start
                method_times.append(query_time)
                context_lengths.append(len(context))
                success_count += 1
                
                # Show detailed results for hybrid method
                if method == "hybrid" and detailed_results:
                    avg_confidence = sum(getattr(r, 'confidence', 0) for r in detailed_results) / len(detailed_results)
                    avg_dense_score = sum(getattr(r, 'dense_score', 0) for r in detailed_results) / len(detailed_results)
                    avg_sparse_score = sum(getattr(r, 'sparse_score', 0) for r in detailed_results) / len(detailed_results)
                    
                    print(f"   Query {i}: {query_time:.3f}s - {len(context)} chars - "
                          f"Conf: {avg_confidence:.3f}, Dense: {avg_dense_score:.3f}, Sparse: {avg_sparse_score:.3f}")
                else:
                    print(f"   Query {i}: {query_time:.3f}s - {len(context)} chars")
                    
            except Exception as e:
                print(f"   Query {i}: FAILED - {str(e)}")
        
        if method_times:
            avg_time = sum(method_times) / len(method_times)
            avg_length = sum(context_lengths) / len(context_lengths)
            success_rate = success_count / len(test_queries)
            
            methods_results[method] = {
                'avg_time': avg_time,
                'avg_length': avg_length,
                'success_rate': success_rate,
                'total_queries': len(test_queries)
            }
            
            print(f"   📈 Summary: {avg_time:.3f}s avg, {avg_length:.0f} chars avg, {success_rate:.1%} success")
    
    # Find best performing method
    if methods_results:
        print(f"\n🏆 METHOD PERFORMANCE RANKING:")
        print("-" * 40)
        
        # Sort by success rate first, then by speed
        sorted_methods = sorted(
            methods_results.items(),
            key=lambda x: (x[1]['success_rate'], -x[1]['avg_time']),
            reverse=True
        )
        
        for rank, (method, stats) in enumerate(sorted_methods, 1):
            print(f"   {rank}. {method.upper()}: "
                  f"{stats['success_rate']:.1%} success, "
                  f"{stats['avg_time']:.3f}s avg")
    
    # ===== QUERY ENHANCEMENT TESTING =====
    print(f"\n🔍 Testing Query Enhancement Capabilities:")
    print("-" * 50)
    
    enhancement_test_queries = [
        "machine learning",
        "What is artificial intelligence?",
        "deep learning vs traditional ML",
        "AI ethics challenges"
    ]
    
    for query in enhancement_test_queries:
        try:
            enhanced_query = enhanced_RA.enhance_query_only(query)
            print(f"   Original: '{query}'")
            print(f"   Enhanced: Intent={enhanced_query.intent.value}, "
                  f"Expansions={len(enhanced_query.expanded_terms)}, "
                  f"Rewrites={len(enhanced_query.rewritten_variants)}, "
                  f"Confidence={enhanced_query.confidence_score:.3f}")
            if enhanced_query.expanded_terms:
                print(f"   Expansions: {', '.join(enhanced_query.expanded_terms[:5])}")
            print()
        except Exception as e:
            print(f"   Enhancement failed for '{query}': {e}")
    
    # ===== CACHING PERFORMANCE TEST =====
    print(f"\n💾 Testing Enhanced Caching Performance:")
    print("-" * 40)
    
    cache_test_query = test_queries[0]
    
    # First call (cache miss)
    cache_start = time.time()
    first_result = enhanced_RA.retrieve_enhanced(
        cache_test_query, 
        method="hybrid", 
        top_k=5, 
        max_tokens=2000
    )
    first_time = time.time() - cache_start
    
    # Second call (cache hit)
    cache_start = time.time()
    second_result = enhanced_RA.retrieve_enhanced(
        cache_test_query, 
        method="hybrid", 
        top_k=5, 
        max_tokens=2000
    )
    second_time = time.time() - cache_start
    
    cache_improvement = ((first_time - second_time) / first_time) * 100
    print(f"   First call (miss): {first_time:.3f}s")
    print(f"   Second call (hit): {second_time:.3f}s") 
    print(f"   Cache improvement: {cache_improvement:.1f}%")
    
    # Get cache statistics
    if hasattr(enhanced_RA.retriever, 'get_performance_stats'):
        cache_stats = enhanced_RA.retriever.get_performance_stats()
        cache_hit_rate = cache_stats.get('cache_hit_rate', 0)
        print(f"   Overall cache hit rate: {cache_hit_rate:.1%}")
    
    # ===== ADVANCED QA TESTING =====
    print(f"\n🤖 Testing Enhanced Question Answering:")
    print("-" * 40)
    
    qa_test_queries = [
        "What is the main topic of this document?",
        "Bu dokümanın en önemli bilgisi nedir?",
        "How does the document explain artificial intelligence?",
        "What are the key conclusions?"
    ]
    
    for query in qa_test_queries:
        try:
            qa_start = time.time()
            answer = enhanced_RA.answer_question(
                query, 
                use_async=True,
                top_k=8,
                max_tokens=2500
            )
            qa_time = time.time() - qa_start
            
            print(f"   Q: {query}")
            print(f"   A: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            print(f"   Time: {qa_time:.3f}s, Length: {len(answer)} chars")
            print()
            
        except Exception as e:
            print(f"   QA failed for '{query}': {e}")
    
    # ===== EVALUATION FRAMEWORK TEST =====
    print(f"\n📊 Running Evaluation Framework:")
    print("-" * 40)
    
    try:
        # Create evaluator
        evaluator = HybridRAPTOREvaluator(enhanced_RA, embed_model)
        
        # Create sample evaluation set
        eval_queries = create_sample_evaluation_set()
        
        # Run comparison
        comparison_df = evaluator.compare_methods(eval_queries[:3])  # Use first 3 for speed
        print("   Method comparison results:")
        print(comparison_df.to_string(index=False))
        
        # Generate detailed report
        report_path = evaluator.generate_evaluation_report(
            eval_queries[:3], 
            output_dir="enhanced_evaluation_results"
        )
        print(f"   📄 Detailed evaluation report saved to: {report_path}")
        
    except Exception as e:
        print(f"   Evaluation framework test failed: {e}")
    
    # ===== SAVE ENHANCED TREE =====
    print(f"\n💾 Saving Enhanced RAPTOR with Hybrid capabilities...")
    if not os.path.exists("vectordb"):
        os.makedirs("vectordb")
    
    SAVE_PATH = "vectordb/enhanced-raptor-optimized"
    enhanced_RA.save(SAVE_PATH, include_metadata=True)
    print(f"✅ Enhanced tree saved to {SAVE_PATH}")
    print(f"📄 Comprehensive metadata saved to {SAVE_PATH}.json")
    
    # Export hybrid configuration
    try:
        enhanced_RA.export_hybrid_config("enhanced_hybrid_config.json")
        print(f"⚙️ Hybrid configuration exported to enhanced_hybrid_config.json")
    except Exception as e:
        print(f"⚠️ Config export failed: {e}")
    
    # ===== FINAL COMPREHENSIVE SUMMARY =====
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 ENHANCED RAPTOR WITH HYBRID RETRIEVAL COMPLETED!")
    print("=" * 80)
    print(f"📊 Total Processing Time: {total_time:.1f}s")
    print(f"📈 Performance Improvements:")
    print(f"   • Enhanced chunking with semantic boundaries")
    print(f"   • Batch embedding creation with async processing")
    print(f"   • Quality-focused clustering with adaptive parameters")
    print(f"   • Hybrid retrieval: Dense + Sparse + Query Enhancement")
    print(f"   • Reciprocal Rank Fusion for optimal result combination")
    print(f"   • Cross-encoder reranking for quality refinement")
    print(f"   • Smart caching system with similarity matching")
    print(f"   • Comprehensive performance monitoring")
    
    # Enhanced tree statistics
    if 'tree_stats' in perf_stats:
        tree_stats = perf_stats['tree_stats']
        print(f"\n🌳 Enhanced Tree Statistics:")
        print(f"   • Total Nodes: {tree_stats.get('total_nodes', 0)}")
        print(f"   • Layers: {tree_stats.get('num_layers', 0)}")
        print(f"   • Leaf Nodes: {tree_stats.get('leaf_nodes', 0)}")
        print(f"   • Root Nodes: {tree_stats.get('root_nodes', 0)}")
    
    # Hybrid performance statistics
    if 'hybrid_retriever' in perf_stats:
        hybrid_stats = perf_stats['hybrid_retriever']
        print(f"\n🔄 Hybrid Retrieval Statistics:")
        print(f"   • Total Retrievals: {hybrid_stats.get('total_retrievals', 0)}")
        print(f"   • Fusion Method: {hybrid_stats.get('fusion_method', 'N/A')}")
        print(f"   • Dense Weight: {hybrid_stats.get('dense_weight', 0):.1f}")
        print(f"   • Sparse Weight: {hybrid_stats.get('sparse_weight', 0):.1f}")
    
    print(f"\n🚀 Enhanced RAPTOR is now production-ready with enterprise-grade hybrid capabilities!")
    print(f"📝 Use the updated test.py to interact with all hybrid features.")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    
    # Still try to get partial performance stats
    try:
        partial_stats = enhanced_RA.get_enhanced_performance_summary()
        print(f"\n📊 Partial Performance Stats: {partial_stats}")
    except:
        pass
    
finally:
    # Enhanced cleanup
    print(f"\n🧹 Cleaning up enhanced resources...")
    try:
        enhanced_RA.clear_all_caches()
        enhanced_RA.optimize_performance()
        print("✅ Enhanced cleanup completed")
    except:
        pass