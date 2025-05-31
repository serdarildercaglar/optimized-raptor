import logging
import os
import hashlib
import time
import asyncio
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from functools import lru_cache
import pickle
from pathlib import Path

import tiktoken
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, AsyncOpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class QueryResult:
    """Cached query result container"""
    query_hash: str
    selected_nodes: List[Node]
    context: str
    similarity_scores: List[float]
    timestamp: float
    retrieval_params: Dict
    layer_information: Optional[List[Dict]] = None


@dataclass 
class RetrievalMetrics:
    """Retrieval performance metrics"""
    total_queries: int = 0
    cache_hits: int = 0
    avg_retrieval_time: float = 0.0
    avg_nodes_retrieved: float = 0.0
    total_retrieval_time: float = 0.0
    layer_usage_stats: Dict[int, int] = field(default_factory=dict)
    similarity_threshold_hits: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_queries, 1)
    
    def update(self, retrieval_time: float, nodes_count: int, cache_hit: bool = False, 
               layers_used: List[int] = None):
        """Update metrics with new retrieval"""
        self.total_queries += 1
        if cache_hit:
            self.cache_hits += 1
        
        self.total_retrieval_time += retrieval_time
        self.avg_retrieval_time = self.total_retrieval_time / self.total_queries
        
        # Update average nodes retrieved
        total_nodes = (self.avg_nodes_retrieved * (self.total_queries - 1)) + nodes_count
        self.avg_nodes_retrieved = total_nodes / self.total_queries
        
        # Update layer usage stats
        if layers_used:
            for layer in layers_used:
                self.layer_usage_stats[layer] = self.layer_usage_stats.get(layer, 0) + 1


class QueryCache:
    """Advanced query result caching with similarity-based matching"""
    
    def __init__(self, cache_dir: str = "query_cache", max_size: int = 1000, 
                 similarity_threshold: float = 0.95, ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        
        # In-memory caches
        self.exact_cache = {}  # Exact query matches
        self.embedding_cache = {}  # Query embeddings for similarity matching
        self.access_times = {}  # LRU tracking
        
    def _get_query_hash(self, query: str, params: Dict) -> str:
        """Generate hash for query + parameters"""
        content = f"{query}:{sorted(params.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self.ttl
    
    def get_exact(self, query: str, params: Dict) -> Optional[QueryResult]:
        """Get exact query match from cache"""
        query_hash = self._get_query_hash(query, params)
        
        if query_hash in self.exact_cache:
            result = self.exact_cache[query_hash]
            if self._is_cache_valid(result.timestamp):
                self.access_times[query_hash] = time.time()
                return result
            else:
                del self.exact_cache[query_hash]
                if query_hash in self.access_times:
                    del self.access_times[query_hash]
        
        return None
    
    async def get_similar(self, query: str, query_embedding: List[float], 
                         params: Dict, embedding_model: BaseEmbeddingModel) -> Optional[QueryResult]:
        """Find similar cached queries using embedding similarity"""
        if not self.embedding_cache:
            return None
        
        best_similarity = 0.0
        best_result = None
        
        for cached_hash, (cached_embedding, cached_result) in self.embedding_cache.items():
            if not self._is_cache_valid(cached_result.timestamp):
                continue
                
            # Check parameter compatibility (exact match for now)
            if cached_result.retrieval_params != params:
                continue
            
            # Calculate similarity
            similarity = 1 - np.linalg.norm(
                np.array(query_embedding) - np.array(cached_embedding)
            )
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_result = cached_result
        
        if best_result:
            logging.debug(f"Similar query cache hit (similarity: {best_similarity:.3f})")
            return best_result
        
        return None
    
    def set(self, query: str, params: Dict, result: QueryResult, 
           query_embedding: Optional[List[float]] = None):
        """Store query result in cache"""
        query_hash = self._get_query_hash(query, params)
        result.query_hash = query_hash
        result.timestamp = time.time()
        result.retrieval_params = params
        
        # Manage cache size (LRU eviction)
        if len(self.exact_cache) >= self.max_size:
            lru_hash = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.exact_cache[lru_hash]
            del self.access_times[lru_hash]
            if lru_hash in self.embedding_cache:
                del self.embedding_cache[lru_hash]
        
        # Store exact match
        self.exact_cache[query_hash] = result
        self.access_times[query_hash] = time.time()
        
        # Store embedding for similarity matching
        if query_embedding:
            self.embedding_cache[query_hash] = (query_embedding, result)
    
    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_hashes = [
            h for h, result in self.exact_cache.items()
            if current_time - result.timestamp > self.ttl
        ]
        
        for hash_key in expired_hashes:
            if hash_key in self.exact_cache:
                del self.exact_cache[hash_key]
            if hash_key in self.access_times:
                del self.access_times[hash_key]
            if hash_key in self.embedding_cache:
                del self.embedding_cache[hash_key]


class AdaptiveRetrieval:
    """Adaptive retrieval strategies based on query characteristics"""
    
    @staticmethod
    def analyze_query(query: str) -> Dict[str, any]:
        """Analyze query characteristics for adaptive parameters"""
        words = query.split()
        
        return {
            'length': len(query),
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'has_questions': any(w in query.lower() for w in ['what', 'how', 'why', 'when', 'where', 'who']),
            'has_specifics': any(w in query.lower() for w in ['specific', 'detail', 'exactly', 'precise']),
            'complexity_score': len(words) * (1 + query.count(',') + query.count(';'))
        }
    
    @staticmethod
    def adapt_retrieval_params(query_analysis: Dict, base_config: Dict) -> Dict:
        """Adapt retrieval parameters based on query analysis"""
        adapted_config = base_config.copy()
        
        # Adjust top_k based on query complexity
        if query_analysis['complexity_score'] > 20:  # Complex query
            adapted_config['top_k'] = min(adapted_config.get('top_k', 5) + 2, 15)
        elif query_analysis['word_count'] <= 3:  # Simple query
            adapted_config['top_k'] = max(adapted_config.get('top_k', 5) - 1, 2)
        
        # Adjust threshold for specific queries
        if query_analysis['has_specifics']:
            adapted_config['threshold'] = adapted_config.get('threshold', 0.5) + 0.1
        
        # Adjust max_tokens for detailed queries
        if query_analysis['has_questions'] and query_analysis['word_count'] > 5:
            adapted_config['max_tokens'] = min(adapted_config.get('max_tokens', 3500) + 500, 5000)
        
        return adapted_config


class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
        # New performance parameters
        enable_caching=None,
        cache_ttl=None,
        similarity_cache_threshold=None,
        adaptive_retrieval=None,
        enable_metrics=None,
        early_termination=None,
        confidence_threshold=None,
        max_concurrent_retrievals=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("o200k_base")
        self.tokenizer = tokenizer

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = AsyncOpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer
        
        # Enhanced performance parameters
        if enable_caching is None:
            enable_caching = True
        self.enable_caching = enable_caching
        
        if cache_ttl is None:
            cache_ttl = 3600  # 1 hour
        self.cache_ttl = cache_ttl
        
        if similarity_cache_threshold is None:
            similarity_cache_threshold = 0.95
        self.similarity_cache_threshold = similarity_cache_threshold
        
        if adaptive_retrieval is None:
            adaptive_retrieval = True
        self.adaptive_retrieval = adaptive_retrieval
        
        if enable_metrics is None:
            enable_metrics = True
        self.enable_metrics = enable_metrics
        
        if early_termination is None:
            early_termination = True
        self.early_termination = early_termination
        
        if confidence_threshold is None:
            confidence_threshold = 0.8
        self.confidence_threshold = confidence_threshold
        
        if max_concurrent_retrievals is None:
            max_concurrent_retrievals = 5
        self.max_concurrent_retrievals = max_concurrent_retrievals

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
            Enable Caching: {enable_caching}
            Cache TTL: {cache_ttl}
            Similarity Cache Threshold: {similarity_cache_threshold}
            Adaptive Retrieval: {adaptive_retrieval}
            Enable Metrics: {enable_metrics}
            Early Termination: {early_termination}
            Confidence Threshold: {confidence_threshold}
            Max Concurrent Retrievals: {max_concurrent_retrievals}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
            enable_caching=self.enable_caching,
            cache_ttl=self.cache_ttl,
            similarity_cache_threshold=self.similarity_cache_threshold,
            adaptive_retrieval=self.adaptive_retrieval,
            enable_metrics=self.enable_metrics,
            early_termination=self.early_termination,
            confidence_threshold=self.confidence_threshold,
            max_concurrent_retrievals=self.max_concurrent_retrievals,
        )
        return config_log


class TreeRetriever(BaseRetriever):

    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model
        
        # Enhanced features
        self.enable_caching = config.enable_caching
        self.adaptive_retrieval = config.adaptive_retrieval
        self.enable_metrics = config.enable_metrics
        self.early_termination = config.early_termination
        self.confidence_threshold = config.confidence_threshold
        
        # Initialize components
        self.query_cache = QueryCache(
            ttl=config.cache_ttl,
            similarity_threshold=config.similarity_cache_threshold
        ) if self.enable_caching else None
        
        self.metrics = RetrievalMetrics() if self.enable_metrics else None
        self.adaptive_strategy = AdaptiveRetrieval() if self.adaptive_retrieval else None
        
        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)
        self.retrieval_semaphore = asyncio.Semaphore(config.max_concurrent_retrievals)

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        """Synchronous embedding creation (backward compatibility)"""
        return self.embedding_model.create_embedding(text)
    
    async def create_embedding_async(self, text: str) -> List[float]:
        """Asynchronous embedding creation"""
        return await self.embedding_model.create_embedding_async(text)

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> Tuple[List[Node], str]:
        """Synchronous collapsed tree retrieval (backward compatibility)"""
        query_embedding = self.create_embedding(query)
        return self._retrieve_collapse_tree_internal(query_embedding, top_k, max_tokens)
    
    async def retrieve_information_collapse_tree_async(self, query: str, top_k: int, max_tokens: int) -> Tuple[List[Node], str]:
        """Asynchronous collapsed tree retrieval"""
        query_embedding = await self.create_embedding_async(query)
        return self._retrieve_collapse_tree_internal(query_embedding, top_k, max_tokens)
    
    def _retrieve_collapse_tree_internal(self, query_embedding: List[float], top_k: int, max_tokens: int) -> Tuple[List[Node], str]:
        """Internal collapsed tree retrieval logic"""
        selected_nodes = []
        node_list = get_node_list(self.tree.all_nodes)
        embeddings = get_embeddings(node_list, self.context_embedding_model)
        distances = distances_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        confidence_scores = []
        
        for idx in indices[:top_k]:
            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                if self.early_termination and confidence_scores:
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    if avg_confidence > self.confidence_threshold:
                        logging.debug(f"Early termination: confidence {avg_confidence:.3f} > {self.confidence_threshold}")
                        break
                else:
                    break

            selected_nodes.append(node)
            total_tokens += node_tokens
            
            # Calculate confidence score (1 - normalized distance)
            confidence = 1 - (distances[idx] / max(distances) if max(distances) > 0 else 0)
            confidence_scores.append(confidence)

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> Tuple[List[Node], str]:
        """Synchronous hierarchical retrieval (backward compatibility)"""
        query_embedding = self.create_embedding(query)
        return self._retrieve_hierarchical_internal(query_embedding, current_nodes, num_layers)
    
    async def retrieve_information_async(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> Tuple[List[Node], str]:
        """Asynchronous hierarchical retrieval"""
        query_embedding = await self.create_embedding_async(query)
        return self._retrieve_hierarchical_internal(query_embedding, current_nodes, num_layers)
    
    def _retrieve_hierarchical_internal(
        self, query_embedding: List[float], current_nodes: List[Node], num_layers: int
    ) -> Tuple[List[Node], str]:
        """Internal hierarchical retrieval logic"""
        selected_nodes = []
        node_list = current_nodes
        layers_used = []

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.context_embedding_model)
            distances = distances_from_embeddings(query_embedding, embeddings)
            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]
            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]
            selected_nodes.extend(nodes_to_add)
            
            # Track layer usage
            if nodes_to_add:
                current_layer = self.tree_node_index_to_layer.get(nodes_to_add[0].index, -1)
                layers_used.append(current_layer)

            if layer != num_layers - 1:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]
                
                # Early termination if no children found
                if not node_list:
                    logging.debug(f"Early termination: no children found at layer {layer}")
                    break

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
        use_async: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Enhanced retrieve with caching and adaptive strategies"""
        if use_async:
            return asyncio.run(self.retrieve_async(
                query, start_layer, num_layers, top_k, max_tokens, 
                collapse_tree, return_layer_information
            ))
        else:
            return self._retrieve_sync(
                query, start_layer, num_layers, top_k, max_tokens,
                collapse_tree, return_layer_information
            )
    
    def _retrieve_sync(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Synchronous retrieve implementation"""
        start_time = time.time()
        
        # Parameter validation
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        # Set defaults
        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        # Validation
        if not isinstance(start_layer, int) or not (0 <= start_layer <= self.tree.num_layers):
            raise ValueError("start_layer must be an integer between 0 and tree.num_layers")
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        # Prepare retrieval parameters
        retrieval_params = {
            'start_layer': start_layer,
            'num_layers': num_layers,
            'top_k': top_k,
            'max_tokens': max_tokens,
            'collapse_tree': collapse_tree,
            'selection_mode': self.selection_mode,
            'threshold': self.threshold
        }
        
        # Check cache first
        cached_result = None
        if self.query_cache:
            cached_result = self.query_cache.get_exact(query, retrieval_params)
        
        if cached_result:
            logging.debug("Cache hit for exact query match")
            retrieval_time = time.time() - start_time
            if self.metrics:
                self.metrics.update(retrieval_time, len(cached_result.selected_nodes), cache_hit=True)
            
            if return_layer_information:
                return cached_result.context, cached_result.layer_information or []
            return cached_result.context
        
        # Adaptive parameter adjustment
        if self.adaptive_strategy:
            query_analysis = self.adaptive_strategy.analyze_query(query)
            retrieval_params = self.adaptive_strategy.adapt_retrieval_params(
                query_analysis, retrieval_params
            )
            # Update local variables with adapted params
            top_k = retrieval_params['top_k']
            max_tokens = retrieval_params['max_tokens']
            logging.debug(f"Adaptive parameters: top_k={top_k}, max_tokens={max_tokens}")

        # Perform retrieval
        if collapse_tree:
            logging.debug("Using collapsed_tree retrieval")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        # Prepare layer information
        layer_information = []
        layers_used = []
        if return_layer_information or self.metrics:
            for node in selected_nodes:
                layer_num = self.tree_node_index_to_layer[node.index]
                layer_info = {
                    "node_index": node.index,
                    "layer_number": layer_num,
                }
                layer_information.append(layer_info)
                layers_used.append(layer_num)

        # Cache the result
        if self.query_cache:
            query_result = QueryResult(
                query_hash="",
                selected_nodes=selected_nodes,
                context=context,
                similarity_scores=[],
                timestamp=time.time(),
                retrieval_params=retrieval_params,
                layer_information=layer_information if return_layer_information else None
            )
            self.query_cache.set(query, retrieval_params, query_result)

        # Update metrics
        retrieval_time = time.time() - start_time
        if self.metrics:
            self.metrics.update(retrieval_time, len(selected_nodes), layers_used=layers_used)
        
        logging.info(
            f"Retrieved {len(selected_nodes)} nodes in {retrieval_time:.3f}s "
            f"(cache_hit_rate: {self.metrics.cache_hit_rate:.2%})" if self.metrics else ""
        )

        if return_layer_information:
            return context, layer_information
        return context
    
    async def retrieve_async(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Fully asynchronous retrieve with advanced caching"""
        async with self.retrieval_semaphore:
            start_time = time.time()
            
            # Parameter setup (same as sync version)
            start_layer = self.start_layer if start_layer is None else start_layer
            num_layers = self.num_layers if num_layers is None else num_layers
            
            retrieval_params = {
                'start_layer': start_layer,
                'num_layers': num_layers,
                'top_k': top_k,
                'max_tokens': max_tokens,
                'collapse_tree': collapse_tree,
                'selection_mode': self.selection_mode,
                'threshold': self.threshold
            }
            
            # Check exact cache first
            cached_result = None
            if self.query_cache:
                cached_result = self.query_cache.get_exact(query, retrieval_params)
            
            if not cached_result and self.query_cache:
                # Check similarity cache
                query_embedding = await self.create_embedding_async(query)
                cached_result = await self.query_cache.get_similar(
                    query, query_embedding, retrieval_params, self.embedding_model
                )
            
            if cached_result:
                retrieval_time = time.time() - start_time
                if self.metrics:
                    self.metrics.update(retrieval_time, len(cached_result.selected_nodes), cache_hit=True)
                
                if return_layer_information:
                    return cached_result.context, cached_result.layer_information or []
                return cached_result.context
            
            # Adaptive parameters
            if self.adaptive_strategy:
                query_analysis = self.adaptive_strategy.analyze_query(query)
                retrieval_params = self.adaptive_strategy.adapt_retrieval_params(
                    query_analysis, retrieval_params
                )
                top_k = retrieval_params['top_k']
                max_tokens = retrieval_params['max_tokens']
            
            # Perform async retrieval
            if collapse_tree:
                selected_nodes, context = await self.retrieve_information_collapse_tree_async(
                    query, top_k, max_tokens
                )
            else:
                layer_nodes = self.tree.layer_to_nodes[start_layer]
                selected_nodes, context = await self.retrieve_information_async(
                    layer_nodes, query, num_layers
                )
            
            # Prepare layer information and cache result
            layer_information = []
            layers_used = []
            if return_layer_information or self.metrics:
                for node in selected_nodes:
                    layer_num = self.tree_node_index_to_layer[node.index]
                    layer_information.append({
                        "node_index": node.index,
                        "layer_number": layer_num,
                    })
                    layers_used.append(layer_num)
            
            # Cache the result with embedding
            if self.query_cache:
                if 'query_embedding' not in locals():
                    query_embedding = await self.create_embedding_async(query)
                
                query_result = QueryResult(
                    query_hash="",
                    selected_nodes=selected_nodes,
                    context=context,
                    similarity_scores=[],
                    timestamp=time.time(),
                    retrieval_params=retrieval_params,
                    layer_information=layer_information if return_layer_information else None
                )
                self.query_cache.set(query, retrieval_params, query_result, query_embedding)
            
            # Update metrics
            retrieval_time = time.time() - start_time
            if self.metrics:
                self.metrics.update(retrieval_time, len(selected_nodes), layers_used=layers_used)
            
            if return_layer_information:
                return context, layer_information
            return context
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.metrics:
            return {}
        
        stats = {
            'total_queries': self.metrics.total_queries,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'avg_retrieval_time': self.metrics.avg_retrieval_time,
            'avg_nodes_retrieved': self.metrics.avg_nodes_retrieved,
            'layer_usage_stats': self.metrics.layer_usage_stats,
            'total_retrieval_time': self.metrics.total_retrieval_time,
        }
        
        if self.query_cache:
            stats.update({
                'cache_size': len(self.query_cache.exact_cache),
                'similarity_cache_size': len(self.query_cache.embedding_cache),
            })
        
        return stats
    
    def clear_cache(self):
        """Clear query cache"""
        if self.query_cache:
            self.query_cache.exact_cache.clear()
            self.query_cache.embedding_cache.clear()
            self.query_cache.access_times.clear()
            logging.info("Query cache cleared")
    
    def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        if self.query_cache:
            self.query_cache.clear_expired()
            logging.info("Expired cache entries cleaned up")