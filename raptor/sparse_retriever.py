# raptor/sparse_retriever.py
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import math
from collections import defaultdict, Counter

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi, BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer

from .Retrievers import BaseRetriever
from .tree_structures import Node
from .utils import split_text

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class SparseRetrievalResult:
    """Container for sparse retrieval results"""
    node: Node
    score: float
    query_terms_matched: List[str]
    doc_length: int
    term_frequencies: Dict[str, int]


class TextPreprocessor:
    """Advanced text preprocessing for sparse retrieval"""
    
    def __init__(self, language: str = "english", custom_stopwords: List[str] = None):
        self.language = language
        self.custom_stopwords = set(custom_stopwords) if custom_stopwords else set()
        
        # Default English stopwords (you can expand this)
        self.default_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        self.stopwords = self.default_stopwords.union(self.custom_stopwords)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing"""
        # Convert to lowercase
        text = text.lower()
        
        # Simple tokenization (you can use more sophisticated tokenizers)
        import re
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stopwords and len(token) > 2]
        
        return tokens
    
    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query with same logic as documents"""
        return self.preprocess_text(query)


class AdvancedBM25Retriever(BaseRetriever):
    """
    Advanced BM25-based sparse retriever with multiple algorithms and caching
    """
    
    def __init__(self, 
                 algorithm: str = "bm25_okapi",  # bm25_okapi, bm25_plus, tfidf
                 k1: float = 1.2,
                 b: float = 0.75,
                 delta: float = 1.0,  # for BM25Plus
                 language: str = "english",
                 enable_caching: bool = True,
                 cache_dir: str = "sparse_cache"):
        
        self.algorithm = algorithm
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.language = language
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir)
        
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.preprocessor = TextPreprocessor(language)
        self.bm25_model = None
        self.tfidf_model = None
        self.corpus_tokens = []
        self.nodes = []
        self.doc_metadata = {}
        
        # Performance tracking
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        logging.info(f"Initialized AdvancedBM25Retriever with algorithm: {algorithm}")
    
    def _get_cache_path(self, content_hash: str) -> Path:
        """Get cache file path for given content hash"""
        return self.cache_dir / f"bm25_index_{content_hash}.pkl"
    
    def _calculate_content_hash(self, nodes: List[Node]) -> str:
        """Calculate hash for node contents"""
        import hashlib
        content = "".join([node.text for node in nodes])
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def build_from_nodes(self, nodes: List[Node], save_cache: bool = True):
        """
        Build sparse index from tree nodes
        
        Args:
            nodes: List of nodes to index
            save_cache: Whether to save index to cache
        """
        start_time = time.time()
        
        # Check cache first
        content_hash = self._calculate_content_hash(nodes)
        cache_path = self._get_cache_path(content_hash)
        
        if self.enable_caching and cache_path.exists():
            try:
                self._load_from_cache(cache_path)
                logging.info(f"Loaded BM25 index from cache: {cache_path}")
                return
            except Exception as e:
                logging.warning(f"Cache load failed: {e}, rebuilding index")
        
        # Build index from scratch
        self.nodes = nodes
        self._build_index()
        
        # Save to cache
        if self.enable_caching and save_cache:
            try:
                self._save_to_cache(cache_path)
                logging.info(f"Saved BM25 index to cache: {cache_path}")
            except Exception as e:
                logging.warning(f"Cache save failed: {e}")
        
        build_time = time.time() - start_time
        logging.info(f"Built sparse index with {len(nodes)} documents in {build_time:.2f}s")
    
    def _build_index(self):
        """Build the sparse retrieval index"""
        # Preprocess all documents
        self.corpus_tokens = []
        self.doc_metadata = {}
        
        for i, node in enumerate(self.nodes):
            tokens = self.preprocessor.preprocess_text(node.text)
            self.corpus_tokens.append(tokens)
            
            # Store metadata
            self.doc_metadata[i] = {
                'node': node,
                'original_text': node.text,
                'token_count': len(tokens),
                'unique_tokens': len(set(tokens)),
                'term_frequencies': Counter(tokens)
            }
        
        # Build retrieval model based on algorithm
        if self.algorithm == "bm25_okapi":
            self.bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        elif self.algorithm == "bm25_plus":
            self.bm25_model = BM25Plus(self.corpus_tokens, k1=self.k1, b=self.b, delta=self.delta)
        elif self.algorithm == "tfidf":
            # Convert tokens back to text for TfidfVectorizer
            corpus_text = [" ".join(tokens) for tokens in self.corpus_tokens]
            self.tfidf_model = TfidfVectorizer(
                tokenizer=lambda x: x.split(),  # Pre-tokenized
                lowercase=False,  # Already lowercased
                token_pattern=None
            )
            self.tfidf_matrix = self.tfidf_model.fit_transform(corpus_text)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _save_to_cache(self, cache_path: Path):
        """Save index to cache"""
        cache_data = {
            'algorithm': self.algorithm,
            'k1': self.k1,
            'b': self.b,
            'delta': self.delta,
            'corpus_tokens': self.corpus_tokens,
            'doc_metadata': self.doc_metadata,
            'nodes': self.nodes
        }
        
        if self.algorithm in ["bm25_okapi", "bm25_plus"]:
            cache_data['bm25_model'] = self.bm25_model
        elif self.algorithm == "tfidf":
            cache_data['tfidf_model'] = self.tfidf_model
            cache_data['tfidf_matrix'] = self.tfidf_matrix
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_from_cache(self, cache_path: Path):
        """Load index from cache"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify compatibility
        if cache_data['algorithm'] != self.algorithm:
            raise ValueError("Cached algorithm doesn't match current algorithm")
        
        self.corpus_tokens = cache_data['corpus_tokens']
        self.doc_metadata = cache_data['doc_metadata']
        self.nodes = cache_data['nodes']
        
        if self.algorithm in ["bm25_okapi", "bm25_plus"]:
            self.bm25_model = cache_data['bm25_model']
        elif self.algorithm == "tfidf":
            self.tfidf_model = cache_data['tfidf_model']
            self.tfidf_matrix = cache_data['tfidf_matrix']
    
    def retrieve(self, query: str, top_k: int = 10, min_score: float = 0.0) -> str:
        """
        Retrieve relevant documents for query (backward compatibility)
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            Concatenated text of retrieved documents
        """
        results = self.retrieve_with_scores(query, top_k, min_score)
        return "\n\n".join([result.node.text for result in results])
    
    def retrieve_with_scores(self, query: str, top_k: int = 10, 
                           min_score: float = 0.0) -> List[SparseRetrievalResult]:
        """
        Retrieve relevant documents with detailed scoring information
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of SparseRetrievalResult objects
        """
        start_time = time.time()
        
        if not self.corpus_tokens:
            raise ValueError("Index not built. Call build_from_nodes() first.")
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_query(query)
        
        if not query_tokens:
            logging.warning("Query produced no valid tokens after preprocessing")
            return []
        
        # Get scores based on algorithm
        if self.algorithm in ["bm25_okapi", "bm25_plus"]:
            scores = self.bm25_model.get_scores(query_tokens)
        elif self.algorithm == "tfidf":
            query_vec = self.tfidf_model.transform([" ".join(query_tokens)])
            scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Create results with detailed information
        results = []
        for i, score in enumerate(scores):
            if score >= min_score:
                # Calculate query term matches
                doc_tokens = set(self.corpus_tokens[i])
                query_terms_matched = [token for token in query_tokens if token in doc_tokens]
                
                result = SparseRetrievalResult(
                    node=self.nodes[i],
                    score=float(score),
                    query_terms_matched=query_terms_matched,
                    doc_length=self.doc_metadata[i]['token_count'],
                    term_frequencies=self.doc_metadata[i]['term_frequencies']
                )
                results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        # Update performance metrics
        self.query_count += 1
        query_time = time.time() - start_time
        self.total_query_time += query_time
        
        logging.debug(f"Sparse retrieval: {len(results)} results in {query_time:.3f}s")
        
        return results
    
    async def retrieve_async(self, query: str, top_k: int = 10, 
                           min_score: float = 0.0) -> List[SparseRetrievalResult]:
        """Async version of retrieve_with_scores"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve_with_scores, query, top_k, min_score)
    
    def get_query_analysis(self, query: str) -> Dict:
        """
        Analyze query and provide insights
        
        Args:
            query: Search query to analyze
            
        Returns:
            Dictionary with query analysis information
        """
        query_tokens = self.preprocessor.preprocess_query(query)
        
        # Find documents containing query terms
        term_doc_frequencies = {}
        for token in set(query_tokens):
            doc_count = sum(1 for tokens in self.corpus_tokens if token in tokens)
            term_doc_frequencies[token] = doc_count
        
        return {
            'original_query': query,
            'processed_tokens': query_tokens,
            'token_count': len(query_tokens),
            'unique_tokens': len(set(query_tokens)),
            'term_document_frequencies': term_doc_frequencies,
            'corpus_coverage': {
                token: freq / len(self.corpus_tokens) 
                for token, freq in term_doc_frequencies.items()
            }
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_query_time = self.total_query_time / max(self.query_count, 1)
        
        return {
            'algorithm': self.algorithm,
            'total_queries': self.query_count,
            'total_query_time': self.total_query_time,
            'avg_query_time': avg_query_time,
            'cache_hits': self.cache_hits,
            'corpus_size': len(self.corpus_tokens),
            'vocab_size': len(set(token for tokens in self.corpus_tokens for token in tokens)) if self.corpus_tokens else 0,
            'avg_doc_length': np.mean([len(tokens) for tokens in self.corpus_tokens]) if self.corpus_tokens else 0
        }
    
    def explain_retrieval(self, query: str, doc_index: int) -> Dict:
        """
        Explain why a specific document was retrieved for a query
        
        Args:
            query: Search query
            doc_index: Index of document to explain
            
        Returns:
            Detailed explanation of retrieval score
        """
        if not self.corpus_tokens or doc_index >= len(self.corpus_tokens):
            raise ValueError("Invalid document index or index not built")
        
        query_tokens = self.preprocessor.preprocess_query(query)
        doc_tokens = self.corpus_tokens[doc_index]
        doc_metadata = self.doc_metadata[doc_index]
        
        # Calculate term-level contributions
        term_contributions = {}
        matched_terms = []
        
        for term in query_tokens:
            if term in doc_tokens:
                matched_terms.append(term)
                tf = doc_metadata['term_frequencies'][term]
                
                if self.algorithm in ["bm25_okapi", "bm25_plus"]:
                    # Simplified BM25 term score calculation
                    doc_freq = sum(1 for tokens in self.corpus_tokens if term in tokens)
                    idf = math.log((len(self.corpus_tokens) - doc_freq + 0.5) / (doc_freq + 0.5))
                    
                    tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(doc_tokens) / np.mean([len(t) for t in self.corpus_tokens])))
                    term_score = idf * tf_component
                    
                    term_contributions[term] = {
                        'term_frequency': tf,
                        'document_frequency': doc_freq,
                        'idf': idf,
                        'tf_component': tf_component,
                        'term_score': term_score
                    }
        
        return {
            'query': query,
            'query_tokens': query_tokens,
            'document_index': doc_index,
            'document_length': len(doc_tokens),
            'matched_terms': matched_terms,
            'match_ratio': len(matched_terms) / len(query_tokens) if query_tokens else 0,
            'term_contributions': term_contributions,
            'document_preview': doc_metadata['original_text'][:200] + "..." if len(doc_metadata['original_text']) > 200 else doc_metadata['original_text']
        }


# Convenience function for creating retriever
def create_sparse_retriever(algorithm: str = "bm25_okapi", **kwargs) -> AdvancedBM25Retriever:
    """
    Create a sparse retriever with specified algorithm
    
    Args:
        algorithm: Retrieval algorithm ("bm25_okapi", "bm25_plus", "tfidf")
        **kwargs: Additional arguments passed to AdvancedBM25Retriever
        
    Returns:
        Configured AdvancedBM25Retriever instance
    """
    return AdvancedBM25Retriever(algorithm=algorithm, **kwargs)