# raptor/hybrid_retriever.py
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from .Retrievers import BaseRetriever
from .tree_retriever import TreeRetriever
from .sparse_retriever import AdvancedBM25Retriever, SparseRetrievalResult
from .query_enhancement import QueryEnhancer, EnhancedQuery
from .tree_structures import Node
from .EmbeddingModels import BaseEmbeddingModel, AsyncOpenAIEmbeddingModel

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class FusionMethod(Enum):
    """Different fusion methods for combining results"""
    RRF = "reciprocal_rank_fusion"      # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"       # Weighted sum of scores
    BORDA_COUNT = "borda_count"         # Borda count method
    CONDORCET = "condorcet"             # Condorcet fusion
    DISTRIBUTIONAL = "distributional"   # Distribution-based fusion


@dataclass
class HybridRetrievalResult:
    """Container for hybrid retrieval results"""
    node: Node
    dense_score: float
    sparse_score: float
    fused_score: float
    rank_dense: int
    rank_sparse: int
    final_rank: int
    query_terms_matched: List[str] = field(default_factory=list)
    rerank_score: Optional[float] = None
    confidence: float = 0.0
    explanation: Dict = field(default_factory=dict)


class ScoreNormalizer:
    """Normalize scores from different retrieval systems"""
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """Min-max normalization"""
        if not scores or min(scores) == max(scores):
            return [0.0] * len(scores)
        
        min_score, max_score = min(scores), max(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """Z-score normalization"""
        if not scores or np.std(scores) == 0:
            return [0.0] * len(scores)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        return [(score - mean_score) / std_score for score in scores]
    
    @staticmethod
    def rank_normalize(scores: List[float]) -> List[float]:
        """Rank-based normalization"""
        if not scores:
            return []
        
        # Convert scores to ranks (higher score = lower rank number)
        score_rank_pairs = [(score, i) for i, score in enumerate(scores)]
        score_rank_pairs.sort(reverse=True)
        
        ranks = [0] * len(scores)
        for rank, (_, original_idx) in enumerate(score_rank_pairs):
            ranks[original_idx] = 1.0 / (rank + 1)  # Reciprocal rank
        
        return ranks


class ResultFusion:
    """Fuse results from multiple retrievers"""
    
    def __init__(self, method: FusionMethod = FusionMethod.RRF):
        self.method = method
        self.normalizer = ScoreNormalizer()
    
    def fuse_results(self, dense_results: List[Tuple[Node, float]], 
                    sparse_results: List[SparseRetrievalResult],
                    dense_weight: float = 0.6, 
                    sparse_weight: float = 0.4) -> List[HybridRetrievalResult]:
        """
        Fuse dense and sparse retrieval results
        
        Args:
            dense_results: List of (node, score) from dense retriever
            sparse_results: List of SparseRetrievalResult from sparse retriever
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            
        Returns:
            List of fused HybridRetrievalResult objects
        """
        # Create unified result mapping
        node_results = {}
        
        # Process dense results
        dense_scores = [score for _, score in dense_results]
        normalized_dense_scores = self.normalizer.min_max_normalize(dense_scores)
        
        for i, (node, score) in enumerate(dense_results):
            node_id = id(node)  # Use object id as unique identifier
            node_results[node_id] = {
                'node': node,
                'dense_score': normalized_dense_scores[i],
                'dense_rank': i + 1,
                'sparse_score': 0.0,
                'sparse_rank': float('inf'),
                'query_terms_matched': []
            }
        
        # Process sparse results
        sparse_scores = [result.score for result in sparse_results]
        normalized_sparse_scores = self.normalizer.min_max_normalize(sparse_scores)
        
        for i, result in enumerate(sparse_results):
            node_id = id(result.node)
            norm_score = normalized_sparse_scores[i]
            
            if node_id in node_results:
                # Update existing entry
                node_results[node_id]['sparse_score'] = norm_score
                node_results[node_id]['sparse_rank'] = i + 1
                node_results[node_id]['query_terms_matched'] = result.query_terms_matched
            else:
                # Create new entry (sparse-only result)
                node_results[node_id] = {
                    'node': result.node,
                    'dense_score': 0.0,
                    'dense_rank': float('inf'),
                    'sparse_score': norm_score,
                    'sparse_rank': i + 1,
                    'query_terms_matched': result.query_terms_matched
                }
        
        # Apply fusion method
        fused_results = []
        
        for node_id, data in node_results.items():
            if self.method == FusionMethod.RRF:
                fused_score = self._reciprocal_rank_fusion(
                    data['dense_rank'], data['sparse_rank']
                )
            elif self.method == FusionMethod.WEIGHTED_SUM:
                fused_score = (dense_weight * data['dense_score'] + 
                              sparse_weight * data['sparse_score'])
            elif self.method == FusionMethod.BORDA_COUNT:
                fused_score = self._borda_count_fusion(
                    data['dense_rank'], data['sparse_rank'], len(node_results)
                )
            else:
                # Default to weighted sum
                fused_score = (dense_weight * data['dense_score'] + 
                              sparse_weight * data['sparse_score'])
            
            result = HybridRetrievalResult(
                node=data['node'],
                dense_score=data['dense_score'],
                sparse_score=data['sparse_score'],
                fused_score=fused_score,
                rank_dense=data['dense_rank'],
                rank_sparse=data['sparse_rank'],
                final_rank=0,  # Will be set after sorting
                query_terms_matched=data['query_terms_matched']
            )
            
            fused_results.append(result)
        
        # Sort by fused score and assign final ranks
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)
        for i, result in enumerate(fused_results):
            result.final_rank = i + 1
        
        return fused_results
    
    def _reciprocal_rank_fusion(self, rank1: float, rank2: float, k: float = 60.0) -> float:
        """Reciprocal Rank Fusion (RRF)"""
        score1 = 1.0 / (k + rank1) if rank1 != float('inf') else 0.0
        score2 = 1.0 / (k + rank2) if rank2 != float('inf') else 0.0
        return score1 + score2
    
    def _borda_count_fusion(self, rank1: float, rank2: float, total_docs: int) -> float:
        """Borda Count fusion"""
        score1 = (total_docs - rank1) if rank1 != float('inf') else 0
        score2 = (total_docs - rank2) if rank2 != float('inf') else 0
        return score1 + score2


class CrossEncoderReranker:
    """Rerank results using cross-encoder or embedding similarity"""
    
    def __init__(self, embedding_model: BaseEmbeddingModel, 
                 rerank_method: str = "embedding_similarity"):
        self.embedding_model = embedding_model
        self.rerank_method = rerank_method
        
        # Performance tracking
        self.rerank_count = 0
        self.total_rerank_time = 0.0
    
    async def rerank_results(self, query: str, 
                           results: List[HybridRetrievalResult],
                           top_k: int = None) -> List[HybridRetrievalResult]:
        """
        Rerank results using query-document similarity
        
        Args:
            query: Original query
            results: List of hybrid results to rerank
            top_k: Number of top results to rerank (None for all)
            
        Returns:
            Reranked list of results
        """
        start_time = time.time()
        
        if not results:
            return results
        
        # Optionally limit reranking to top results for efficiency
        results_to_rerank = results[:top_k] if top_k else results
        results_not_reranked = results[top_k:] if top_k else []
        
        if self.rerank_method == "embedding_similarity":
            reranked = await self._rerank_by_embedding_similarity(query, results_to_rerank)
        else:
            # Fallback to no reranking
            reranked = results_to_rerank
        
        # Combine reranked and not-reranked results
        final_results = reranked + results_not_reranked
        
        # Update final ranks
        for i, result in enumerate(final_results):
            result.final_rank = i + 1
        
        # Update metrics
        self.rerank_count += 1
        rerank_time = time.time() - start_time
        self.total_rerank_time += rerank_time
        
        logging.debug(f"Reranked {len(results_to_rerank)} results in {rerank_time:.3f}s")
        
        return final_results
    
    async def _rerank_by_embedding_similarity(self, query: str, 
                                            results: List[HybridRetrievalResult]) -> List[HybridRetrievalResult]:
        """Rerank using query-document embedding similarity"""
        if not results:
            return results
        
        # Get query embedding
        query_embedding = await self.embedding_model.create_embedding_async(query)
        
        # Calculate similarities
        for result in results:
            # Get document embedding (assuming it exists in node)
            if hasattr(result.node, 'embeddings') and self.embedding_model.__class__.__name__ in result.node.embeddings:
                doc_embedding = result.node.embeddings[self.embedding_model.__class__.__name__]
            else:
                # Fallback: create embedding for document text
                doc_embedding = await self.embedding_model.create_embedding_async(result.node.text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            result.rerank_score = float(similarity)
            
            # Update confidence based on multiple signals
            result.confidence = self._calculate_confidence(result)
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        
        return results
    
    def _calculate_confidence(self, result: HybridRetrievalResult) -> float:
        """Calculate confidence score for a result"""
        confidence_factors = []
        
        # Factor 1: Rerank score
        if result.rerank_score is not None:
            confidence_factors.append(result.rerank_score)
        
        # Factor 2: Fusion score
        confidence_factors.append(result.fused_score)
        
        # Factor 3: Query term matching
        if result.query_terms_matched:
            match_factor = min(len(result.query_terms_matched) / 5.0, 1.0)  # Max boost for 5+ matches
            confidence_factors.append(match_factor)
        
        # Factor 4: Rank consistency
        rank_consistency = 1.0 / (1.0 + abs(result.rank_dense - result.rank_sparse))
        confidence_factors.append(rank_consistency)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0


class HybridRetriever(BaseRetriever):
    """
    Main hybrid retriever that combines dense, sparse, and query enhancement
    """
    
    def __init__(self, 
                 dense_retriever: TreeRetriever,
                 sparse_retriever: AdvancedBM25Retriever,
                 query_enhancer: QueryEnhancer = None,
                 fusion_method: FusionMethod = FusionMethod.RRF,
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4,
                 enable_reranking: bool = True,
                 rerank_top_k: int = 20):
        
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.query_enhancer = query_enhancer
        
        # Fusion configuration
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Reranking configuration
        self.enable_reranking = enable_reranking
        self.rerank_top_k = rerank_top_k
        
        # Initialize components
        self.result_fusion = ResultFusion(fusion_method)
        
        if enable_reranking:
            # Use the same embedding model as dense retriever if available
            embedding_model = getattr(dense_retriever, 'embedding_model', AsyncOpenAIEmbeddingModel())
            self.reranker = CrossEncoderReranker(embedding_model)
        else:
            self.reranker = None
        
        # Performance tracking
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0
        self.enhancement_time = 0.0
        self.fusion_time = 0.0
        self.rerank_time = 0.0
        
        logging.info(f"Initialized HybridRetriever with fusion method: {fusion_method.value}")
    
    def retrieve(self, query: str, top_k: int = 10, max_tokens: int = 3500,
                enhance_query: bool = True, **kwargs) -> str:
        """
        Synchronous hybrid retrieval (backward compatibility)
        
        Args:
            query: Search query
            top_k: Number of top results to return
            max_tokens: Maximum tokens in response
            enhance_query: Whether to enhance the query
            **kwargs: Additional arguments passed to retrievers
            
        Returns:
            Concatenated text of retrieved documents
        """
        # Run async version synchronously
        import asyncio
        results = asyncio.run(self.retrieve_hybrid_async(
            query, top_k, max_tokens, enhance_query, **kwargs
        ))
        
        # Return concatenated text
        return "\n\n".join([result.node.text for result in results])
    
    async def retrieve_hybrid_async(self, query: str, top_k: int = 10, 
                                  max_tokens: int = 3500, enhance_query: bool = True,
                                  **kwargs) -> List[HybridRetrievalResult]:
        """
        Asynchronous hybrid retrieval with full pipeline
        
        Args:
            query: Search query
            top_k: Number of top results to return
            max_tokens: Maximum tokens in response
            enhance_query: Whether to enhance the query
            **kwargs: Additional arguments
            
        Returns:
            List of HybridRetrievalResult objects
        """
        start_time = time.time()
        self.retrieval_count += 1
        
        # Step 1: Query Enhancement (optional)
        enhanced_query = None
        if enhance_query and self.query_enhancer:
            enhancement_start = time.time()
            enhanced_query = await self.query_enhancer.enhance_query(query)
            self.enhancement_time += time.time() - enhancement_start
            
            # Use enhanced query for retrieval
            search_queries = [enhanced_query.normalized] + enhanced_query.rewritten_variants
            
            logging.debug(f"Enhanced query: {enhanced_query.intent.value} intent, "
                         f"{len(enhanced_query.expanded_terms)} expansions")
        else:
            search_queries = [query]
        
        # Step 2: Dense Retrieval
        dense_results = []
        for search_query in search_queries[:2]:  # Limit to 2 variants for efficiency
            try:
                if hasattr(self.dense_retriever, 'retrieve_async'):
                    context = await self.dense_retriever.retrieve_async(
                        search_query, top_k=top_k * 2, max_tokens=max_tokens, **kwargs
                    )
                else:
                    # Fallback to sync
                    context = self.dense_retriever.retrieve(
                        search_query, top_k=top_k * 2, max_tokens=max_tokens, **kwargs
                    )
                
                # Extract nodes and scores (this is a simplified approach)
                # In practice, you'd need to modify TreeRetriever to return structured results
                dense_results.extend(self._extract_dense_results(context, search_query))
                
            except Exception as e:
                logging.warning(f"Dense retrieval failed for query '{search_query}': {e}")
        
        # Step 3: Sparse Retrieval
        sparse_results = []
        for search_query in search_queries[:2]:  # Limit to 2 variants for efficiency
            try:
                sparse_batch = await self.sparse_retriever.retrieve_async(
                    search_query, top_k=top_k * 2
                )
                sparse_results.extend(sparse_batch)
            except Exception as e:
                logging.warning(f"Sparse retrieval failed for query '{search_query}': {e}")
        
        # Step 4: Fusion
        fusion_start = time.time()
        fused_results = self.result_fusion.fuse_results(
            dense_results, sparse_results, self.dense_weight, self.sparse_weight
        )
        self.fusion_time += time.time() - fusion_start
        
        # Step 5: Reranking (optional)
        if self.enable_reranking and self.reranker:
            rerank_start = time.time()
            fused_results = await self.reranker.rerank_results(
                query, fused_results, self.rerank_top_k
            )
            self.rerank_time += time.time() - rerank_start
        
        # Step 6: Filter by token limit and return top_k
        final_results = self._filter_by_tokens(fused_results, max_tokens, top_k)
        
        # Update metrics
        total_time = time.time() - start_time
        self.total_retrieval_time += total_time
        
        logging.info(f"Hybrid retrieval: {len(final_results)} results in {total_time:.3f}s "
                    f"(dense: {len(dense_results)}, sparse: {len(sparse_results)})")
        
        return final_results
    
    def _extract_dense_results(self, context: str, query: str) -> List[Tuple[Node, float]]:
        """
        Extract nodes and scores from dense retrieval context
        This is a placeholder - you'd need to modify TreeRetriever to return structured results
        """
        # This is a simplified approach - in practice you'd modify TreeRetriever
        # to return List[Tuple[Node, float]] instead of concatenated text
        
        # For now, create dummy results
        # You should modify TreeRetriever.retrieve() to return structured results
        return []
    
    def _filter_by_tokens(self, results: List[HybridRetrievalResult], 
                         max_tokens: int, top_k: int) -> List[HybridRetrievalResult]:
        """Filter results by token limit"""
        import tiktoken
        tokenizer = tiktoken.get_encoding("o200k_base")
        
        filtered_results = []
        total_tokens = 0
        
        for result in results:
            if len(filtered_results) >= top_k:
                break
            
            node_tokens = len(tokenizer.encode(result.node.text))
            if total_tokens + node_tokens <= max_tokens:
                filtered_results.append(result)
                total_tokens += node_tokens
            elif not filtered_results:  # Include at least one result
                filtered_results.append(result)
                break
        
        return filtered_results
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        avg_retrieval_time = self.total_retrieval_time / max(self.retrieval_count, 1)
        
        stats = {
            'total_retrievals': self.retrieval_count,
            'total_retrieval_time': self.total_retrieval_time,
            'avg_retrieval_time': avg_retrieval_time,
            'enhancement_time': self.enhancement_time,
            'fusion_time': self.fusion_time,
            'rerank_time': self.rerank_time,
            'fusion_method': self.fusion_method.value,
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'reranking_enabled': self.enable_reranking,
        }
        
        # Add component stats
        if hasattr(self.dense_retriever, 'get_performance_stats'):
            stats['dense_retriever'] = self.dense_retriever.get_performance_stats()
        
        if hasattr(self.sparse_retriever, 'get_performance_stats'):
            stats['sparse_retriever'] = self.sparse_retriever.get_performance_stats()
        
        if self.query_enhancer and hasattr(self.query_enhancer, 'get_performance_stats'):
            stats['query_enhancer'] = self.query_enhancer.get_performance_stats()
        
        if self.reranker:
            avg_rerank_time = self.reranker.total_rerank_time / max(self.reranker.rerank_count, 1)
            stats['reranker'] = {
                'total_reranks': self.reranker.rerank_count,
                'avg_rerank_time': avg_rerank_time
            }
        
        return stats


# Convenience functions
def create_hybrid_retriever(dense_retriever: TreeRetriever,
                          sparse_retriever: AdvancedBM25Retriever,
                          query_enhancer: QueryEnhancer = None,
                          **kwargs) -> HybridRetriever:
    """
    Create a hybrid retriever with specified components
    
    Args:
        dense_retriever: Dense retriever (TreeRetriever)
        sparse_retriever: Sparse retriever (BM25)
        query_enhancer: Query enhancer (optional)
        **kwargs: Additional configuration
        
    Returns:
        Configured HybridRetriever instance
    """
    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        query_enhancer=query_enhancer,
        **kwargs
    )