# hybrid_raptor_demo.py
"""
Enhanced RAPTOR Hybrid Retrieval Demo

This demo shows how to use the new hybrid retrieval capabilities:
- Dense + Sparse retrieval fusion
- Query enhancement and expansion  
- Advanced result reranking
- Performance comparison and optimization
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import enhanced RAPTOR components
from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, 
    HybridConfig, 
    create_enhanced_raptor
)
from raptor.hybrid_retriever import FusionMethod

def setup_enhanced_raptor():
    """Setup Enhanced RAPTOR with optimized configuration"""
    print("🚀 Setting up Enhanced RAPTOR with Hybrid Capabilities...")
    
    # Standard RAPTOR configuration
    embed_model = CustomEmbeddingModel()
    sum_model = GPT41SummarizationModel()
    
    raptor_config = RetrievalAugmentationConfig(
        # Optimized tree building
        tb_max_tokens=100,
        tb_summarization_length=300,
        tb_num_layers=4,
        tb_batch_size=100,
        tb_build_mode="async",
        
        # Enhanced retrieval
        tr_enable_caching=True,
        tr_adaptive_retrieval=True,
        tr_early_termination=True,
        
        # Models
        summarization_model=sum_model,
        embedding_model=embed_model,
        
        # Performance
        enable_async=True,
        enable_caching=True,
        enable_metrics=True,
        performance_monitoring=True
    )
    
    # Hybrid configuration
    hybrid_config = HybridConfig(
        # Core hybrid features
        enable_hybrid=True,
        enable_query_enhancement=True,
        enable_sparse_retrieval=True,
        enable_reranking=True,
        
        # Fusion settings
        fusion_method=FusionMethod.RRF,  # Reciprocal Rank Fusion
        dense_weight=0.6,               # 60% dense, 40% sparse
        sparse_weight=0.4,
        
        # Sparse retrieval (BM25 Okapi)
        sparse_algorithm="bm25_okapi",
        sparse_k1=1.2,
        sparse_b=0.75,
        
        # Query enhancement
        max_query_expansions=5,
        semantic_expansion=True,
        
        # Reranking
        rerank_top_k=20,
        
        # Performance
        enable_caching=True,
        cache_dir="hybrid_cache"
    )
    
    print(f"✅ Configuration complete:")
    print(f"   • Fusion Method: {hybrid_config.fusion_method.value}")
    print(f"   • Dense Weight: {hybrid_config.dense_weight}")
    print(f"   • Sparse Algorithm: {hybrid_config.sparse_algorithm}")
    print(f"   • Query Enhancement: {hybrid_config.enable_query_enhancement}")
    print(f"   • Reranking: {hybrid_config.enable_reranking}")
    
    return raptor_config, hybrid_config


def load_or_create_enhanced_raptor(text_file: str = "data.txt", 
                                  tree_path: str = "vectordb/raptor-optimized"):
    """Load existing tree or create new Enhanced RAPTOR"""
    
    raptor_config, hybrid_config = setup_enhanced_raptor()
    
    # Try to load existing tree first
    if Path(tree_path).exists():
        print(f"📂 Loading existing tree from {tree_path}...")
        try:
            enhanced_raptor = EnhancedRetrievalAugmentation(
                config=raptor_config,
                tree=tree_path,
                hybrid_config=hybrid_config
            )
            print("✅ Tree loaded successfully with hybrid enhancements!")
            return enhanced_raptor
        except Exception as e:
            print(f"⚠️ Failed to load tree: {e}")
            print("📝 Creating new tree...")
    
    # Create new tree
    print(f"🔨 Building new Enhanced RAPTOR tree...")
    
    # Load text data
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Loaded {len(text):,} characters from {text_file}")
    except FileNotFoundError:
        print(f"❌ Text file {text_file} not found. Using sample text.")
        text = """
        Artificial Intelligence represents one of the most significant technological 
        advancements of our time. Machine learning algorithms enable computers to learn 
        from data without explicit programming. Deep learning, a subset of machine learning,
        uses neural networks with multiple layers to analyze complex patterns in data.
        
        Natural Language Processing (NLP) allows machines to understand and generate 
        human language. Recent developments in transformer architectures have led to 
        breakthrough models like GPT and BERT. These models demonstrate remarkable 
        capabilities in text generation, translation, and comprehension.
        
        Computer vision enables machines to interpret and analyze visual information.
        Convolutional Neural Networks (CNNs) have revolutionized image recognition tasks.
        Applications range from medical diagnosis to autonomous vehicles.
        
        The future of AI holds immense potential across various industries including
        healthcare, finance, education, and transportation. However, ethical considerations
        and responsible development remain crucial challenges to address.
        """
    
    # Create Enhanced RAPTOR
    enhanced_raptor = create_enhanced_raptor(
        text=text,
        config=raptor_config,
        hybrid_config=hybrid_config
    )
    
    # Save the tree
    enhanced_raptor.save(tree_path, include_metadata=True)
    print(f"💾 Enhanced tree saved to {tree_path}")
    
    return enhanced_raptor


def demo_query_enhancement(enhanced_raptor):
    """Demonstrate query enhancement capabilities"""
    print("\n" + "="*80)
    print("🔍 QUERY ENHANCEMENT DEMONSTRATION")
    print("="*80)
    
    test_queries = [
        "What is AI?",
        "machine learning algorithms",
        "How does deep learning work?",
        "Applications of computer vision",
        "Future of artificial intelligence"
    ]
    
    for query in test_queries:
        print(f"\n📝 Original Query: '{query}'")
        print("-" * 50)
        
        try:
            enhanced_query = enhanced_raptor.enhance_query_only(query)
            
            print(f"🎯 Intent: {enhanced_query.intent.value} (confidence: {enhanced_query.confidence_score:.2f})")
            print(f"📋 Normalized: '{enhanced_query.normalized}'")
            print(f"🏷️ Entities: {enhanced_query.key_entities}")
            print(f"🔄 Expansions: {enhanced_query.expanded_terms}")
            
            if enhanced_query.rewritten_variants:
                print(f"✏️ Rewrites: {enhanced_query.rewritten_variants}")
            
            print(f"⚡ Processing Time: {enhanced_query.processing_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Enhancement failed: {e}")


def demo_retrieval_comparison(enhanced_raptor):
    """Compare different retrieval methods"""
    print("\n" + "="*80)
    print("🏁 RETRIEVAL METHOD COMPARISON")
    print("="*80)
    
    test_queries = [
        "What are the applications of machine learning?",
        "Explain deep learning neural networks",
        "computer vision CNN",
        "AI ethics challenges"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        try:
            comparison = enhanced_raptor.compare_retrieval_methods(query, top_k=3)
            
            for method, results in comparison['results'].items():
                if 'error' in results:
                    print(f"{method.upper():>10}: ❌ {results['error']}")
                else:
                    print(f"{method.upper():>10}: "
                          f"{results['result_count']} results, "
                          f"{results['retrieval_time']:.3f}s, "
                          f"{results['context_length']} chars")
                    
                    # Show confidence for hybrid method
                    if method == 'hybrid' and 'avg_confidence' in results:
                        print(f"{'':>12}Confidence: {results['avg_confidence']:.3f}")
        
        except Exception as e:
            print(f"❌ Comparison failed: {e}")


def demo_hybrid_optimization(enhanced_raptor):
    """Demonstrate hybrid parameter optimization"""
    print("\n" + "="*80)
    print("⚙️ HYBRID PARAMETER OPTIMIZATION")
    print("="*80)
    
    test_queries = [
        "machine learning definition",
        "neural network architecture", 
        "AI applications healthcare",
        "deep learning vs traditional ML",
        "computer vision techniques"
    ]
    
    print(f"🧪 Testing {len(test_queries)} queries for parameter optimization...")
    
    try:
        optimization_results = enhanced_raptor.optimize_hybrid_parameters(test_queries)
        
        print(f"✅ Optimization completed!")
        print(f"📊 Tested {len(optimization_results['tested_parameters'])} parameter combinations")
        
        # Show best parameters
        if optimization_results['best_parameters']:
            best = optimization_results['best_parameters']
            print(f"\n🏆 Best Parameters:")
            print(f"   Dense Weight: {best['dense_weight']}")
            print(f"   Sparse Weight: {best['sparse_weight']}")
            print(f"   Performance Score: {best['performance_score']:.3f}")
        
        # Show all tested combinations
        print(f"\n📈 All Tested Combinations:")
        for params in optimization_results['tested_parameters']:
            print(f"   Dense: {params['dense_weight']}, "
                  f"Sparse: {params['sparse_weight']} → "
                  f"Score: {params['performance_score']:.3f} "
                  f"(Time: {params['avg_time']:.3f}s)")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")


def demo_performance_analysis(enhanced_raptor):
    """Analyze performance improvements"""
    print("\n" + "="*80)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Get comprehensive performance summary
    performance = enhanced_raptor.get_enhanced_performance_summary()
    
    print("🏗️ Tree Building Performance:")
    if 'tree_builder' in performance:
        builder = performance['tree_builder']
        print(f"   Build Time: {builder.get('total_build_time', 0):.2f}s")
        print(f"   Nodes Created: {builder.get('nodes_created', 0)}")
        print(f"   Avg Layer Time: {builder.get('avg_layer_time', 0):.2f}s")
    
    print("\n🔍 Retrieval Performance:")
    if 'hybrid_retriever' in performance:
        retriever = performance['hybrid_retriever']
        print(f"   Total Queries: {retriever.get('total_retrievals', 0)}")
        print(f"   Avg Query Time: {retriever.get('avg_retrieval_time', 0):.3f}s")
        print(f"   Fusion Method: {retriever.get('fusion_method', 'N/A')}")
        print(f"   Dense Weight: {retriever.get('dense_weight', 0):.1f}")
        print(f"   Sparse Weight: {retriever.get('sparse_weight', 0):.1f}")
    
    print("\n💾 Caching Performance:")
    if 'sparse_retriever' in performance:
        sparse = performance['sparse_retriever']
        print(f"   Sparse Queries: {sparse.get('total_queries', 0)}")
        print(f"   Avg Sparse Time: {sparse.get('avg_query_time', 0):.3f}s")
        print(f"   Vocabulary Size: {sparse.get('vocab_size', 0)}")
    
    print("\n🔄 Query Enhancement:")
    if 'query_enhancer' in performance:
        enhancer = performance['query_enhancer']
        print(f"   Enhancements: {enhancer.get('total_enhancements', 0)}")
        print(f"   Avg Enhancement Time: {enhancer.get('avg_enhancement_time', 0):.3f}s")
        print(f"   Vocabulary Size: {enhancer.get('vocabulary_size', 0)}")
    
    # Hybrid-specific metrics
    print("\n🚀 Hybrid Features:")
    hybrid_features = performance.get('hybrid_features', {})
    for feature, enabled in hybrid_features.items():
        status = "✅" if enabled else "❌"
        print(f"   {feature}: {status}")


def demo_qa_with_explanations(enhanced_raptor):
    """Demonstrate Q&A with detailed explanations"""
    print("\n" + "="*80)
    print("🤖 QUESTION ANSWERING WITH EXPLANATIONS")
    print("="*80)
    
    qa_queries = [
        "What is the main difference between machine learning and deep learning?",
        "How are CNNs used in computer vision?",
        "What are the main challenges in AI ethics?"
    ]
    
    for query in qa_queries:
        print(f"\n❓ Question: {query}")
        print("-" * 70)
        
        try:
            # Get detailed retrieval results
            context, detailed_results = enhanced_raptor.retrieve_enhanced(
                query, 
                method="hybrid", 
                top_k=3,
                return_detailed=True
            )
            
            print(f"📄 Retrieved Context ({len(context)} chars):")
            print(f"   {context[:200]}...")
            
            # Show retrieval details
            if detailed_results:
                print(f"\n🔍 Retrieval Details:")
                for i, result in enumerate(detailed_results[:2], 1):
                    if hasattr(result, 'fused_score'):
                        print(f"   Result {i}: Dense={result.dense_score:.3f}, "
                              f"Sparse={result.sparse_score:.3f}, "
                              f"Fused={result.fused_score:.3f}")
                        if result.query_terms_matched:
                            print(f"              Matched: {result.query_terms_matched}")
            
            # Generate answer
            answer = enhanced_raptor.answer_question(query, max_tokens=2000)
            print(f"\n💡 Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Q&A failed: {e}")


def save_demo_results(enhanced_raptor, output_dir: str = "demo_results"):
    """Save demo results and configurations"""
    print(f"\n💾 Saving demo results to {output_dir}/...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Export configuration
        config_file = output_path / "hybrid_config.json"
        enhanced_raptor.export_hybrid_config(str(config_file))
        
        # Export performance summary
        performance_file = output_path / "performance_summary.json"
        performance = enhanced_raptor.get_enhanced_performance_summary()
        with open(performance_file, 'w') as f:
            json.dump(performance, f, indent=2, default=str)
        
        print(f"✅ Results saved:")
        print(f"   • Configuration: {config_file}")
        print(f"   • Performance: {performance_file}")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")


def main():
    """Main demo function"""
    print("🎯 ENHANCED RAPTOR HYBRID RETRIEVAL DEMO")
    print("=" * 80)
    print("This demo showcases the new hybrid retrieval capabilities:")
    print("• Dense + Sparse retrieval fusion")
    print("• Query enhancement and expansion")
    print("• Advanced result reranking")
    print("• Performance optimization")
    print("=" * 80)
    
    try:
        # Setup Enhanced RAPTOR
        enhanced_raptor = load_or_create_enhanced_raptor()
        
        # Run demonstrations
        demo_query_enhancement(enhanced_raptor)
        demo_retrieval_comparison(enhanced_raptor)
        demo_hybrid_optimization(enhanced_raptor)
        demo_performance_analysis(enhanced_raptor)
        demo_qa_with_explanations(enhanced_raptor)
        
        # Save results
        save_demo_results(enhanced_raptor)
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("✨ Key Benefits Demonstrated:")
        print("• 🚀 Improved retrieval quality with hybrid fusion")
        print("• 🧠 Smart query enhancement and expansion")
        print("• 🎯 Better result ranking and relevance")
        print("• ⚡ Optimized performance with caching")
        print("• 📊 Comprehensive monitoring and analytics")
        print("\n🔧 Ready for production use!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# Migration Guide from Standard RAPTOR
def migration_example():
    """
    Example showing how to migrate from standard RAPTOR to Enhanced RAPTOR
    """
    print("\n" + "="*60)
    print("📋 MIGRATION GUIDE: Standard → Enhanced RAPTOR")
    print("="*60)
    
    print("""
    # OLD: Standard RAPTOR
    from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    
    config = RetrievalAugmentationConfig(...)
    RA = RetrievalAugmentation(config=config, tree="path/to/tree")
    result = RA.retrieve("query")
    
    # NEW: Enhanced RAPTOR with Hybrid Features
    from raptor.enhanced_retrieval_augmentation import (
        EnhancedRetrievalAugmentation, HybridConfig
    )
    
    # Same standard config + hybrid config
    hybrid_config = HybridConfig(
        enable_hybrid=True,
        enable_query_enhancement=True,
        fusion_method=FusionMethod.RRF
    )
    
    # Enhanced instance (backward compatible!)
    enhanced_RA = EnhancedRetrievalAugmentation(
        config=config, 
        tree="path/to/tree",
        hybrid_config=hybrid_config
    )
    
    # Standard retrieval (same as before)
    standard_result = enhanced_RA.retrieve("query")
    
    # NEW: Hybrid retrieval with enhancements
    hybrid_result = enhanced_RA.retrieve_enhanced("query", method="hybrid")
    
    # NEW: Query analysis
    analysis = enhanced_RA.analyze_query("query")
    
    # NEW: Method comparison
    comparison = enhanced_RA.compare_retrieval_methods("query")
    """)
    
    print("🔄 Migration is seamless - existing code continues to work!")
    print("➕ New features are opt-in and don't break existing functionality")


# Performance Benchmarking
def benchmark_example():
    """Example of benchmarking different retrieval methods"""
    print("\n" + "="*60)
    print("🏁 BENCHMARKING EXAMPLE")
    print("="*60)
    
    print("""
    # Benchmark different methods
    import time
    
    def benchmark_retrieval_methods(enhanced_raptor, queries):
        methods = ['dense', 'sparse', 'hybrid']
        results = {}
        
        for method in methods:
            start_time = time.time()
            total_results = 0
            
            for query in queries:
                context = enhanced_raptor.retrieve_enhanced(query, method=method)
                total_results += len(context.split())
            
            total_time = time.time() - start_time
            
            results[method] = {
                'total_time': total_time,
                'avg_time_per_query': total_time / len(queries),
                'total_results': total_results
            }
        
        return results
    
    # Usage
    test_queries = ["AI definition", "machine learning", "neural networks"]
    benchmark = benchmark_retrieval_methods(enhanced_raptor, test_queries)
    
    for method, stats in benchmark.items():
        print(f"{method}: {stats['avg_time_per_query']:.3f}s per query")
    """)


if __name__ == "__main__":
    main()
    migration_example()
    benchmark_example()