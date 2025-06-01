# raptor/enhanced_retrieval_augmentation.py
import logging
import asyncio
import time
from typing import List, Dict, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from .hybrid_retriever import HybridRetriever, create_hybrid_retriever, FusionMethod, HybridRetrievalResult
from .sparse_retriever import AdvancedBM25Retriever, create_sparse_retriever
from .query_enhancement import QueryEnhancer, create_query_enhancer, EnhancedQuery
from .tree_retriever import TreeRetriever
from .tree_structures import Tree

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval features"""
    enable_hybrid: bool = True
    enable_query_enhancement: bool = True
    enable_sparse_retrieval: bool = True
    enable_reranking: bool = True
    
    # Fusion settings
    fusion_method: FusionMethod = FusionMethod.RRF
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    
    # Sparse retrieval settings
    sparse_algorithm: str = "bm25_okapi"  # bm25_okapi, bm25_plus, tfidf
    sparse_k1: float = 1.2
    sparse_b: float = 0.75
    
    # Query enhancement settings
    max_query_expansions: int = 5
    semantic_expansion: bool = True
    
    # Reranking settings
    rerank_top_k: int = 20
    
    # Performance settings
    enable_caching: bool = True
    cache_dir: str = "hybrid_cache"


class EnhancedRetrievalAugmentation(RetrievalAugmentation):
    """
    Enhanced RAPTOR with hybrid retrieval capabilities
    
    This extends the original RetrievalAugmentation with:
    - Hybrid dense + sparse retrieval
    - Query enhancement and expansion
    - Advanced result fusion and reranking
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config=None, tree=None, hybrid_config: HybridConfig = None):
        """
        Initialize Enhanced RAPTOR with hybrid capabilities
        
        Args:
            config: Standard RetrievalAugmentationConfig
            tree: Tree instance or path to pickled tree
            hybrid_config: HybridConfig for enhanced features
        """
        # Initialize base RAPTOR
        super().__init__(config, tree)
        
        # Initialize hybrid configuration
        self.hybrid_config = hybrid_config or HybridConfig()
        
        # Enhanced components (initialized after tree is available)
        self.sparse_retriever = None
        self.query_enhancer = None
        self.hybrid_retriever = None
        
        # Performance tracking for hybrid features
        self.hybrid_metrics = {
            'hybrid_queries': 0,
            'query_enhancements': 0,
            'sparse_retrievals': 0,
            'total_hybrid_time': 0.0,
            'fusion_improvements': []
        }
        
        # Initialize hybrid components if tree is available
        if self.tree is not None:
            self._initialize_hybrid_components()
        
        logging.info("Enhanced RAPTOR initialized with hybrid capabilities")
    
    def _initialize_hybrid_components(self):
        """Initialize hybrid retrieval components"""
        if not self.tree:
            return
        
        try:
            # Initialize sparse retriever
            if self.hybrid_config.enable_sparse_retrieval:
                self.sparse_retriever = create_sparse_retriever(
                    algorithm=self.hybrid_config.sparse_algorithm,
                    k1=self.hybrid_config.sparse_k1,
                    b=self.hybrid_config.sparse_b,
                    enable_caching=self.hybrid_config.enable_caching,
                    cache_dir=self.hybrid_config.cache_dir
                )
                
                # Build sparse index from tree nodes
                all_nodes = list(self.tree.all_nodes.values())
                self.sparse_retriever.build_from_nodes(all_nodes)
                
                logging.info(f"Initialized sparse retriever with {len(all_nodes)} nodes")
            
            # Initialize query enhancer
            if self.hybrid_config.enable_query_enhancement:
                embedding_model = getattr(self.retriever, 'embedding_model', None)
                corpus_nodes = list(self.tree.all_nodes.values())
                
                self.query_enhancer = create_query_enhancer(
                    embedding_model=embedding_model,
                    corpus_nodes=corpus_nodes
                )
                
                logging.info("Initialized query enhancer")
            
            # Initialize hybrid retriever
            if self.hybrid_config.enable_hybrid and self.sparse_retriever:
                self.hybrid_retriever = create_hybrid_retriever(
                    dense_retriever=self.retriever,
                    sparse_retriever=self.sparse_retriever,
                    query_enhancer=self.query_enhancer,
                    fusion_method=self.hybrid_config.fusion_method,
                    dense_weight=self.hybrid_config.dense_weight,
                    sparse_weight=self.hybrid_config.sparse_weight,
                    enable_reranking=self.hybrid_config.enable_reranking,
                    rerank_top_k=self.hybrid_config.rerank_top_k
                )
                
                logging.info("Initialized hybrid retriever")
                
        except Exception as e:
            logging.error(f"Failed to initialize hybrid components: {e}")
            # Fallback to standard retrieval
            self.hybrid_config.enable_hybrid = False
    
    def add_documents(self, docs: str, progress_callback: Optional[Callable] = None):
        """
        Enhanced document addition with hybrid component initialization
        """
        # Call parent method
        super().add_documents(docs, progress_callback)
        
        # Initialize hybrid components after tree is built
        self._initialize_hybrid_components()
    
    def retrieve_enhanced(self, 
                         query: str,
                         method: str = "hybrid",  # "hybrid", "dense", "sparse"
                         top_k: int = 10,
                         max_tokens: int = 3500,
                         enhance_query: bool = True,
                         return_detailed: bool = False,
                         **kwargs) -> Union[str, Tuple[str, List[HybridRetrievalResult]]]:
        """
        Enhanced retrieval with multiple methods
        
        Args:
            query: Search query
            method: Retrieval method ("hybrid", "dense", "sparse")
            top_k: Number of top results
            max_tokens: Maximum tokens in response
            enhance_query: Whether to enhance query
            return_detailed: Return detailed results with scores
            **kwargs: Additional arguments
            
        Returns:
            Context string or (context, detailed_results) if return_detailed=True
        """
        start_time = time.time()
        
        if method == "hybrid" and self.hybrid_retriever:
            # Use hybrid retrieval
            results = asyncio.run(self.hybrid_retriever.retrieve_hybrid_async(
                query, top_k, max_tokens, enhance_query, **kwargs
            ))
            context = "\n\n".join([result.node.text for result in results])
            
            self.hybrid_metrics['hybrid_queries'] += 1
            
        elif method == "sparse" and self.sparse_retriever:
            # Use sparse retrieval only
            sparse_results = asyncio.run(self.sparse_retriever.retrieve_async(
                query, top_k
            ))
            context = "\n\n".join([result.node.text for result in sparse_results])
            results = sparse_results  # For return_detailed
            
            self.hybrid_metrics['sparse_retrievals'] += 1
            
        else:
            # Fallback to standard dense retrieval
            context = self.retrieve(query, top_k=top_k, max_tokens=max_tokens, **kwargs)
            results = []  # No detailed results for standard retrieval
        
        # Update metrics
        retrieval_time = time.time() - start_time
        self.hybrid_metrics['total_hybrid_time'] += retrieval_time
        
        if return_detailed:
            return context, results
        else:
            return context
    
    def enhance_query_only(self, query: str) -> EnhancedQuery:
        """
        Enhance query without retrieval (for analysis purposes)
        
        Args:
            query: Query to enhance
            
        Returns:
            EnhancedQuery object with all enhancements
        """
        if not self.query_enhancer:
            raise ValueError("Query enhancer not initialized. Enable query enhancement in hybrid_config.")
        
        enhanced_query = asyncio.run(self.query_enhancer.enhance_query(query))
        self.hybrid_metrics['query_enhancements'] += 1
        
        return enhanced_query
    
    def analyze_query(self, query: str) -> Dict:
        """
        Comprehensive query analysis
        
        Args:
            query: Query to analyze
            
        Returns:
            Analysis results including intent, entities, expansions, etc.
        """
        analysis = {'original_query': query}
        
        # Query enhancement analysis
        if self.query_enhancer:
            enhanced = self.enhance_query_only(query)
            analysis.update({
                'enhanced_query': {
                    'normalized': enhanced.normalized,
                    'intent': enhanced.intent.value,
                    'confidence': enhanced.confidence_score,
                    'entities': enhanced.key_entities,
                    'expansions': enhanced.expanded_terms,
                    'rewrites': enhanced.rewritten_variants,
                    'query_type': enhanced.query_type
                }
            })
        
        # Sparse retrieval analysis
        if self.sparse_retriever:
            sparse_analysis = self.sparse_retriever.get_query_analysis(query)
            analysis['sparse_analysis'] = sparse_analysis
        
        return analysis
    
    def compare_retrieval_methods(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare different retrieval methods for the same query
        
        Args:
            query: Query to test
            top_k: Number of results per method
            
        Returns:
            Comparison results
        """
        comparison = {'query': query, 'results': {}}
        
        # Test each method
        methods = ['dense', 'sparse', 'hybrid']
        
        for method in methods:
            try:
                start_time = time.time()
                context, results = self.retrieve_enhanced(
                    query, method=method, top_k=top_k, return_detailed=True
                )
                retrieval_time = time.time() - start_time
                
                comparison['results'][method] = {
                    'context_length': len(context),
                    'result_count': len(results),
                    'retrieval_time': retrieval_time,
                    'context_preview': context[:200] + "..." if len(context) > 200 else context
                }
                
                # Add method-specific metrics
                if method == 'hybrid' and results:
                    avg_confidence = np.mean([r.confidence for r in results if hasattr(r, 'confidence')])
                    comparison['results'][method]['avg_confidence'] = avg_confidence
                    
            except Exception as e:
                comparison['results'][method] = {'error': str(e)}
        
        return comparison
    
    def get_enhanced_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary including hybrid features
        """
        # Get base performance summary
        base_summary = super().get_performance_summary()
        
        # Add hybrid metrics
        hybrid_summary = {
            'hybrid_features': {
                'enabled': self.hybrid_config.enable_hybrid,
                'query_enhancement_enabled': self.hybrid_config.enable_query_enhancement,
                'sparse_retrieval_enabled': self.hybrid_config.enable_sparse_retrieval,
                'reranking_enabled': self.hybrid_config.enable_reranking
            },
            'hybrid_metrics': self.hybrid_metrics.copy()
        }
        
        # Add component performance stats
        if self.hybrid_retriever:
            hybrid_summary['hybrid_retriever'] = self.hybrid_retriever.get_performance_stats()
        
        if self.sparse_retriever:
            hybrid_summary['sparse_retriever'] = self.sparse_retriever.get_performance_stats()
        
        if self.query_enhancer:
            hybrid_summary['query_enhancer'] = self.query_enhancer.get_performance_stats()
        
        # Combine summaries
        base_summary.update(hybrid_summary)
        
        return base_summary
    
    def optimize_hybrid_parameters(self, test_queries: List[str], 
                                 ground_truth: List[str] = None) -> Dict:
        """
        Optimize hybrid retrieval parameters using test queries
        
        Args:
            test_queries: List of test queries
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            Optimization results and recommended parameters
        """
        if not self.hybrid_retriever:
            raise ValueError("Hybrid retrieval not enabled")
        
        optimization_results = {
            'test_queries': len(test_queries),
            'tested_parameters': [],
            'best_parameters': None,
            'performance_improvements': {}
        }
        
        # Test different weight combinations
        weight_combinations = [
            (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.4, 0.6), (0.3, 0.7)
        ]
        
        best_performance = 0.0
        best_params = None
        
        for dense_w, sparse_w in weight_combinations:
            # Update weights
            self.hybrid_retriever.dense_weight = dense_w
            self.hybrid_retriever.sparse_weight = sparse_w
            
            # Test performance
            total_time = 0.0
            total_results = 0
            
            for query in test_queries:
                start_time = time.time()
                context, results = self.retrieve_enhanced(
                    query, method="hybrid", return_detailed=True
                )
                query_time = time.time() - start_time
                
                total_time += query_time
                total_results += len(results)
            
            # Calculate performance score (you can enhance this with relevance metrics)
            avg_time = total_time / len(test_queries)
            avg_results = total_results / len(test_queries)
            performance_score = avg_results / avg_time  # Simple score
            
            optimization_results['tested_parameters'].append({
                'dense_weight': dense_w,
                'sparse_weight': sparse_w,
                'avg_time': avg_time,
                'avg_results': avg_results,
                'performance_score': performance_score
            })
            
            if performance_score > best_performance:
                best_performance = performance_score
                best_params = (dense_w, sparse_w)
        
        # Set best parameters
        if best_params:
            self.hybrid_retriever.dense_weight = best_params[0]
            self.hybrid_retriever.sparse_weight = best_params[1]
            
            optimization_results['best_parameters'] = {
                'dense_weight': best_params[0],
                'sparse_weight': best_params[1],
                'performance_score': best_performance
            }
        
        logging.info(f"Parameter optimization completed. Best weights: {best_params}")
        
        return optimization_results
    
    def export_hybrid_config(self, filepath: str):
        """Export current hybrid configuration to file"""
        import json
        
        config_dict = {
            'hybrid_config': {
                'enable_hybrid': self.hybrid_config.enable_hybrid,
                'enable_query_enhancement': self.hybrid_config.enable_query_enhancement,
                'enable_sparse_retrieval': self.hybrid_config.enable_sparse_retrieval,
                'enable_reranking': self.hybrid_config.enable_reranking,
                'fusion_method': self.hybrid_config.fusion_method.value,
                'dense_weight': self.hybrid_config.dense_weight,
                'sparse_weight': self.hybrid_config.sparse_weight,
                'sparse_algorithm': self.hybrid_config.sparse_algorithm,
                'max_query_expansions': self.hybrid_config.max_query_expansions,
                'rerank_top_k': self.hybrid_config.rerank_top_k
            },
            'performance_summary': self.get_enhanced_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logging.info(f"Hybrid configuration exported to {filepath}")


# Enhanced Configuration Class
class EnhancedRetrievalAugmentationConfig(RetrievalAugmentationConfig):
    """
    Enhanced configuration that includes hybrid features
    """
    
    def __init__(self, hybrid_config: HybridConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid_config = hybrid_config or HybridConfig()


# Convenience functions
def create_enhanced_raptor(text: str = None, 
                          config: RetrievalAugmentationConfig = None,
                          hybrid_config: HybridConfig = None,
                          tree_path: str = None) -> EnhancedRetrievalAugmentation:
    """
    Create an enhanced RAPTOR instance with hybrid capabilities
    
    Args:
        text: Text to build tree from (if not loading from tree_path)
        config: Standard RAPTOR configuration
        hybrid_config: Hybrid feature configuration
        tree_path: Path to existing tree (alternative to text)
        
    Returns:
        Configured EnhancedRetrievalAugmentation instance
    """
    # Load existing tree or create new one
    if tree_path:
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=config, 
            tree=tree_path, 
            hybrid_config=hybrid_config
        )
    else:
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=config, 
            hybrid_config=hybrid_config
        )
        
        if text:
            enhanced_raptor.add_documents(text)
    
    return enhanced_raptor


# Example usage
def example_usage():
    """Example of how to use Enhanced RAPTOR"""
    from raptor import GPT41SummarizationModel
    from raptor.EmbeddingModels import CustomEmbeddingModel
    
    # Create standard RAPTOR config
    embed_model = CustomEmbeddingModel()
    sum_model = GPT41SummarizationModel()
    
    config = RetrievalAugmentationConfig(
        tb_max_tokens=100,
        tb_summarization_length=200,
        tb_num_layers=3,
        summarization_model=sum_model,
        embedding_model=embed_model,
        enable_async=True
    )
    
    # Create hybrid config
    hybrid_config = HybridConfig(
        enable_hybrid=True,
        enable_query_enhancement=True,
        enable_sparse_retrieval=True,
        enable_reranking=True,
        fusion_method=FusionMethod.RRF,
        dense_weight=0.6,
        sparse_weight=0.4
    )
    
    # Create enhanced RAPTOR
    enhanced_raptor = EnhancedRetrievalAugmentation(
        config=config,
        hybrid_config=hybrid_config
    )
    
    # Add documents
    text = "Your document text here..."
    enhanced_raptor.add_documents(text)
    
    # Use different retrieval methods
    query = "What is the main topic?"
    
    # Standard retrieval
    standard_result = enhanced_raptor.retrieve(query)
    
    # Hybrid retrieval
    hybrid_result = enhanced_raptor.retrieve_enhanced(query, method="hybrid")
    
    # Sparse-only retrieval
    sparse_result = enhanced_raptor.retrieve_enhanced(query, method="sparse")
    
    # Query analysis
    analysis = enhanced_raptor.analyze_query(query)
    
    # Method comparison
    comparison = enhanced_raptor.compare_retrieval_methods(query)
    
    # Performance summary
    performance = enhanced_raptor.get_enhanced_performance_summary()
    
    print("Enhanced RAPTOR example completed!")
    return enhanced_raptor


if __name__ == "__main__":
    example_usage()