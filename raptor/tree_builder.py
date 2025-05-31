import copy
import logging
import os
import asyncio
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, AsyncOpenAIEmbeddingModel, create_embeddings_parallel
from .SummarizationModels import (BaseSummarizationModel,
                                  GPT3TurboSummarizationModel)
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BuildMode(Enum):
    """Build mode for tree construction"""
    SYNC = "sync"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class BuildProgress:
    """Progress tracking for tree building"""
    total_chunks: int = 0
    processed_chunks: int = 0
    total_layers: int = 0
    current_layer: int = 0
    total_nodes: int = 0
    created_nodes: int = 0
    start_time: float = field(default_factory=time.time)
    layer_times: List[float] = field(default_factory=list)
    embedding_time: float = 0.0
    summarization_time: float = 0.0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def chunk_progress(self) -> float:
        return self.processed_chunks / max(self.total_chunks, 1)
    
    @property
    def layer_progress(self) -> float:
        return self.current_layer / max(self.total_layers, 1)
    
    @property
    def node_progress(self) -> float:
        return self.created_nodes / max(self.total_nodes, 1)
    
    def log_progress(self):
        """Log current progress"""
        logging.info(
            f"Progress - Layer: {self.current_layer}/{self.total_layers} "
            f"({self.layer_progress:.1%}), "
            f"Nodes: {self.created_nodes}/{self.total_nodes} "
            f"({self.node_progress:.1%}), "
            f"Time: {self.elapsed_time:.1f}s"
        )


class AsyncProgressCallback:
    """Async callback for progress updates"""
    
    def __init__(self, callback: Optional[Callable[[BuildProgress], None]] = None):
        self.callback = callback
    
    async def update(self, progress: BuildProgress):
        """Update progress asynchronously"""
        if self.callback:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(progress)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.callback, progress)


class AsyncSummarizationWrapper:
    """Wrapper to make summarization models async-compatible"""
    
    def __init__(self, summarization_model: BaseSummarizationModel):
        self.model = summarization_model
    
    async def summarize_async(self, context: str, max_tokens: int = 150) -> str:
        """Async summarization with thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.model.summarize, context, max_tokens
        )
    
    async def summarize_batch(self, contexts: List[str], max_tokens: int = 150, 
                            max_concurrent: int = 5) -> List[str]:
        """Batch summarization with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def summarize_with_semaphore(context: str) -> str:
            async with semaphore:
                return await self.summarize_async(context, max_tokens)
        
        tasks = [summarize_with_semaphore(context) for context in contexts]
        return await asyncio.gather(*tasks)


class TreeBuilderConfig:
    def __init__(
        self,
        tokenizer=None,
        max_tokens=None,
        num_layers=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        summarization_length=None,
        summarization_model=None,
        embedding_models=None,
        cluster_embedding_model=None,
        # New async/performance parameters
        build_mode=None,
        batch_size=None,
        max_concurrent_embeddings=None,
        max_concurrent_summarizations=None,
        enable_progress_tracking=None,
        embedding_cache_enabled=None,
        performance_monitoring=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("o200k_base")
        self.tokenizer = tokenizer

        if max_tokens is None:
            max_tokens = 100
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")
        self.max_tokens = max_tokens

        if num_layers is None:
            num_layers = 5
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")
        self.num_layers = num_layers

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode must be either 'top_k' or 'threshold'")
        self.selection_mode = selection_mode

        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        if summarization_model is None:
            summarization_model = GPT3TurboSummarizationModel()
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        self.summarization_model = summarization_model

        if embedding_models is None:
            embedding_models = {"OpenAI": AsyncOpenAIEmbeddingModel()}
        if not isinstance(embedding_models, dict):
            raise ValueError(
                "embedding_models must be a dictionary of model_name: instance pairs"
            )
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError(
                    "All embedding models must be an instance of BaseEmbeddingModel"
                )
        self.embedding_models = embedding_models

        if cluster_embedding_model is None:
            cluster_embedding_model = "OpenAI"
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError(
                "cluster_embedding_model must be a key in the embedding_models dictionary"
            )
        self.cluster_embedding_model = cluster_embedding_model
        
        # New performance parameters
        if build_mode is None:
            build_mode = BuildMode.ASYNC
        self.build_mode = build_mode
        
        if batch_size is None:
            batch_size = 100
        self.batch_size = batch_size
        
        if max_concurrent_embeddings is None:
            max_concurrent_embeddings = 10
        self.max_concurrent_embeddings = max_concurrent_embeddings
        
        if max_concurrent_summarizations is None:
            max_concurrent_summarizations = 5
        self.max_concurrent_summarizations = max_concurrent_summarizations
        
        if enable_progress_tracking is None:
            enable_progress_tracking = True
        self.enable_progress_tracking = enable_progress_tracking
        
        if embedding_cache_enabled is None:
            embedding_cache_enabled = True
        self.embedding_cache_enabled = embedding_cache_enabled
        
        if performance_monitoring is None:
            performance_monitoring = True
        self.performance_monitoring = performance_monitoring

    def log_config(self):
        config_log = """
        TreeBuilderConfig:
            Tokenizer: {tokenizer}
            Max Tokens: {max_tokens}
            Num Layers: {num_layers}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Summarization Length: {summarization_length}
            Summarization Model: {summarization_model}
            Embedding Models: {embedding_models}
            Cluster Embedding Model: {cluster_embedding_model}
            Build Mode: {build_mode}
            Batch Size: {batch_size}
            Max Concurrent Embeddings: {max_concurrent_embeddings}
            Max Concurrent Summarizations: {max_concurrent_summarizations}
            Progress Tracking: {enable_progress_tracking}
            Performance Monitoring: {performance_monitoring}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
            build_mode=self.build_mode.value if isinstance(self.build_mode, BuildMode) else self.build_mode,
            batch_size=self.batch_size,
            max_concurrent_embeddings=self.max_concurrent_embeddings,
            max_concurrent_summarizations=self.max_concurrent_summarizations,
            enable_progress_tracking=self.enable_progress_tracking,
            performance_monitoring=self.performance_monitoring,
        )
        return config_log


class TreeBuilder:
    """
    Enhanced TreeBuilder with async capabilities and batch processing
    """

    def __init__(self, config) -> None:
        """Initializes the TreeBuilder with enhanced configuration"""
        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model
        
        # Enhanced configuration
        self.build_mode = config.build_mode
        self.batch_size = config.batch_size
        self.max_concurrent_embeddings = config.max_concurrent_embeddings
        self.max_concurrent_summarizations = config.max_concurrent_summarizations
        self.enable_progress_tracking = config.enable_progress_tracking
        self.performance_monitoring = config.performance_monitoring
        
        # Initialize async wrappers
        self.async_summarizer = AsyncSummarizationWrapper(self.summarization_model)
        
        # Progress tracking
        self.progress = BuildProgress() if self.enable_progress_tracking else None
        self.progress_callback = AsyncProgressCallback()
        
        # Performance metrics
        self.metrics = {
            'total_build_time': 0.0,
            'embedding_batch_count': 0,
            'summarization_batch_count': 0,
            'cache_hit_rate': 0.0,
            'layer_times': []
        } if self.performance_monitoring else None

        logging.info(
            f"Successfully initialized TreeBuilder with Config {config.log_config()}"
        )

    def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        """Synchronous node creation (backward compatibility)"""
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))
    
    async def create_node_async(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None,
        precomputed_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[int, Node]:
        """Asynchronous node creation with optional precomputed embeddings"""
        if children_indices is None:
            children_indices = set()
        
        if precomputed_embeddings:
            embeddings = precomputed_embeddings
        else:
            # Create embeddings for all models concurrently
            embedding_tasks = {}
            for model_name, model in self.embedding_models.items():
                embedding_tasks[model_name] = model.create_embedding_async(text)
            
            embeddings = {}
            for model_name, task in embedding_tasks.items():
                embeddings[model_name] = await task
        
        return (index, Node(text, index, children_indices, embeddings))
    
    async def create_nodes_batch(
        self, texts_with_indices: List[Tuple[int, str, Optional[Set[int]]]]
    ) -> Dict[int, Node]:
        """Create multiple nodes in batch with optimized embedding generation"""
        start_time = time.time()
        
        # Extract texts for batch embedding
        texts = [item[1] for item in texts_with_indices]
        
        # Create embeddings for all models in parallel batches
        embedding_results = await create_embeddings_parallel(
            self.embedding_models, texts, self.batch_size
        )
        
        # Create nodes with precomputed embeddings
        nodes = {}
        tasks = []
        
        for i, (index, text, children_indices) in enumerate(texts_with_indices):
            # Prepare precomputed embeddings for this text
            precomputed_embeddings = {
                model_name: embeddings[i] 
                for model_name, embeddings in embedding_results.items()
            }
            
            task = self.create_node_async(index, text, children_indices, precomputed_embeddings)
            tasks.append(task)
        
        # Create all nodes concurrently
        node_results = await asyncio.gather(*tasks)
        
        for index, node in node_results:
            nodes[index] = node
        
        # Update metrics
        if self.metrics:
            self.metrics['embedding_batch_count'] += 1
            
        # Update progress
        if self.progress:
            self.progress.created_nodes += len(nodes)
            self.progress.embedding_time += time.time() - start_time
            
        logging.info(f"Created {len(nodes)} nodes in batch ({time.time() - start_time:.2f}s)")
        
        return nodes

    def create_embedding(self, text) -> List[float]:
        """Synchronous embedding creation (backward compatibility)"""
        return self.embedding_models[self.cluster_embedding_model].create_embedding(text)
    
    async def create_embedding_async(self, text: str) -> List[float]:
        """Asynchronous embedding creation"""
        model = self.embedding_models[self.cluster_embedding_model]
        return await model.create_embedding_async(text)

    def summarize(self, context, max_tokens=150) -> str:
        """Synchronous summarization (backward compatibility)"""
        return self.summarization_model.summarize(context, max_tokens)
    
    async def summarize_async(self, context: str, max_tokens: int = 150) -> str:
        """Asynchronous summarization"""
        return await self.async_summarizer.summarize_async(context, max_tokens)

    def get_relevant_nodes(self, current_node, list_nodes) -> List[Node]:
        """Synchronous relevant nodes (backward compatibility)"""
        embeddings = get_embeddings(list_nodes, self.cluster_embedding_model)
        distances = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]
        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]
        return nodes_to_add

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        """Enhanced multithreaded leaf node creation with progress tracking"""
        start_time = time.time()
        
        if self.progress:
            self.progress.total_chunks = len(chunks)
            self.progress.processed_chunks = 0
        
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node
                
                if self.progress:
                    self.progress.processed_chunks += 1
                    if self.progress.processed_chunks % 10 == 0:
                        self.progress.log_progress()

        if self.metrics:
            self.metrics['layer_times'].append(time.time() - start_time)
            
        logging.info(f"Created {len(leaf_nodes)} leaf nodes in {time.time() - start_time:.2f}s")
        return leaf_nodes
    
    async def async_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        """Asynchronous leaf node creation with batch optimization"""
        start_time = time.time()
        
        if self.progress:
            self.progress.total_chunks = len(chunks)
            self.progress.total_nodes = len(chunks)
        
        # Prepare data for batch creation
        texts_with_indices = [(i, text, None) for i, text in enumerate(chunks)]
        
        # Create nodes in batch
        leaf_nodes = await self.create_nodes_batch(texts_with_indices)
        
        if self.metrics:
            self.metrics['layer_times'].append(time.time() - start_time)
        
        logging.info(f"Created {len(leaf_nodes)} leaf nodes async in {time.time() - start_time:.2f}s")
        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True, 
                       progress_callback: Optional[Callable] = None) -> Tree:
        """Enhanced build from text with mode selection"""
        if self.build_mode == BuildMode.ASYNC:
            # Run async version
            return asyncio.run(self.build_from_text_async(text, progress_callback))
        else:
            # Use original sync version with enhancements
            return self._build_from_text_sync(text, use_multithreading, progress_callback)
    
    def _build_from_text_sync(self, text: str, use_multithreading: bool = True,
                             progress_callback: Optional[Callable] = None) -> Tree:
        """Enhanced synchronous build with progress tracking"""
        start_time = time.time()
        
        if self.progress:
            self.progress = BuildProgress()
            self.progress.start_time = start_time
        
        if progress_callback:
            self.progress_callback = AsyncProgressCallback(progress_callback)
        
        chunks = split_text(text, self.tokenizer, self.max_tokens, enhanced=True)
        logging.info("Creating Leaf Nodes")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks):
                __, node = self.create_node(index, text)
                leaf_nodes[index] = node
                
                if self.progress:
                    self.progress.processed_chunks += 1

        layer_to_nodes = {0: list(leaf_nodes.values())}
        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building All Nodes")
        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)
        
        if self.metrics:
            self.metrics['total_build_time'] = time.time() - start_time
            logging.info(f"Build completed in {self.metrics['total_build_time']:.2f}s")
        
        return tree
    
    async def build_from_text_async(self, text: str, 
                                  progress_callback: Optional[Callable] = None) -> Tree:
        """Fully asynchronous build from text"""
        start_time = time.time()
        
        if self.progress:
            self.progress = BuildProgress()
            self.progress.start_time = start_time
        
        if progress_callback:
            self.progress_callback = AsyncProgressCallback(progress_callback)
        
        # Enhanced text splitting
        chunks = split_text(text, self.tokenizer, self.max_tokens, enhanced=True)
        logging.info(f"Creating {len(chunks)} Leaf Nodes Async")

        # Create leaf nodes asynchronously
        leaf_nodes = await self.async_create_leaf_nodes(chunks)
        
        layer_to_nodes = {0: list(leaf_nodes.values())}
        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings Async")

        logging.info("Building Tree Layers Async")
        all_nodes = copy.deepcopy(leaf_nodes)
        
        # Async tree construction
        root_nodes = await self.construct_tree_async(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)
        
        if self.metrics:
            self.metrics['total_build_time'] = time.time() - start_time
            logging.info(f"Async build completed in {self.metrics['total_build_time']:.2f}s")
        
        return tree

    @abstractmethod
    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer.
        To be implemented by subclasses.
        """
        pass
    
    async def construct_tree_async(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node], 
        layer_to_nodes: Dict[int, List[Node]],
    ) -> Dict[int, Node]:
        """
        Async tree construction. To be implemented by subclasses.
        """
        # Default implementation falls back to sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.construct_tree, current_level_nodes, all_tree_nodes, layer_to_nodes
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.metrics:
            return {}
        
        total_time = self.metrics.get('total_build_time', 0)
        layer_times = self.metrics.get('layer_times', [])
        
        return {
            **self.metrics,
            'avg_layer_time': sum(layer_times) / len(layer_times) if layer_times else 0,
            'embedding_efficiency': self.metrics.get('embedding_batch_count', 0) / max(total_time, 0.1),
            'nodes_per_second': self.progress.created_nodes / max(total_time, 0.1) if self.progress else 0,
        }
    
    def set_progress_callback(self, callback: Callable):
        """Set progress callback for real-time updates"""
        self.progress_callback = AsyncProgressCallback(callback)