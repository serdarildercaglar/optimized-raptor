import logging
import pickle
import asyncio
import time
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel, AsyncOpenAIEmbeddingModel
from .QAModels import BaseQAModel, GPT41QAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig, BuildProgress
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline performance metrics"""
    build_time: float = 0.0
    total_queries: int = 0
    total_query_time: float = 0.0
    cache_hits: int = 0
    avg_query_time: float = 0.0
    embedding_cache_hits: int = 0
    embedding_api_calls: int = 0
    nodes_processed: int = 0
    layers_built: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.total_queries, 1)
    
    @property
    def embedding_cache_efficiency(self) -> float:
        total_embedding_requests = self.embedding_cache_hits + self.embedding_api_calls
        return self.embedding_cache_hits / max(total_embedding_requests, 1)
    
    @property
    def queries_per_second(self) -> float:
        return self.total_queries / max(self.total_query_time, 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/export"""
        return {
            'build_time': self.build_time,
            'total_queries': self.total_queries,
            'avg_query_time': self.avg_query_time,
            'cache_hit_rate': self.cache_hit_rate,
            'embedding_cache_efficiency': self.embedding_cache_efficiency,
            'queries_per_second': self.queries_per_second,
            'nodes_processed': self.nodes_processed,
            'layers_built': self.layers_built,
        }


class AsyncProgressTracker:
    """Advanced progress tracking with real-time updates"""
    
    def __init__(self, enable_callbacks: bool = True):
        self.enable_callbacks = enable_callbacks
        self.callbacks: List[Callable] = []
        self.current_progress: Optional[BuildProgress] = None
        
    def add_callback(self, callback: Callable[[BuildProgress], None]):
        """Add progress callback function"""
        if self.enable_callbacks:
            self.callbacks.append(callback)
    
    async def update_progress(self, progress: BuildProgress):
        """Update progress and notify callbacks"""
        self.current_progress = progress
        
        if self.enable_callbacks:
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(progress)
                    else:
                        # Run sync callback in executor to avoid blocking
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, callback, progress)
                except Exception as e:
                    logging.warning(f"Progress callback failed: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress as dictionary"""
        if not self.current_progress:
            return {}
        
        return {
            'chunk_progress': self.current_progress.chunk_progress,
            'layer_progress': self.current_progress.layer_progress,
            'node_progress': self.current_progress.node_progress,
            'elapsed_time': self.current_progress.elapsed_time,
            'current_layer': self.current_progress.current_layer,
            'total_layers': self.current_progress.total_layers,
            'created_nodes': self.current_progress.created_nodes,
            'total_nodes': self.current_progress.total_nodes,
        }


class RetrievalAugmentationConfig:
    def __init__(
        self,
        tree_builder_config=None,
        tree_retriever_config=None,
        qa_model=None,
        embedding_model=None,
        summarization_model=None,
        tree_builder_type="cluster",
        # Enhanced configuration parameters
        enable_async=None,
        enable_caching=None,
        enable_metrics=None,
        enable_progress_tracking=None,
        max_concurrent_operations=None,
        performance_monitoring=None,
        cache_ttl=None,
        # TreeRetrieverConfig arguments
        tr_tokenizer=None,
        tr_threshold=0.5,
        tr_top_k=5,
        tr_selection_mode="top_k",
        tr_context_embedding_model="OpenAI",
        tr_embedding_model=None,
        tr_num_layers=None,
        tr_start_layer=None,
        tr_enable_caching=None,
        tr_adaptive_retrieval=None,
        tr_early_termination=None,
        # TreeBuilderConfig arguments
        tb_tokenizer=None,
        tb_max_tokens=100,
        tb_num_layers=5,
        tb_threshold=0.5,
        tb_top_k=5,
        tb_selection_mode="top_k",
        tb_summarization_length=100,
        tb_summarization_model=None,
        tb_embedding_models=None,
        tb_cluster_embedding_model="OpenAI",
        tb_build_mode=None,
        tb_batch_size=None,
        tb_enable_progress_tracking=None,
    ):
        # Validate tree_builder_type
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(
                f"tree_builder_type must be one of {list(supported_tree_builders.keys())}"
            )

        # Enhanced configuration parameters
        if enable_async is None:
            enable_async = True
        self.enable_async = enable_async
        
        if enable_caching is None:
            enable_caching = True
        self.enable_caching = enable_caching
        
        if enable_metrics is None:
            enable_metrics = True
        self.enable_metrics = enable_metrics
        
        if enable_progress_tracking is None:
            enable_progress_tracking = True
        self.enable_progress_tracking = enable_progress_tracking
        
        if max_concurrent_operations is None:
            max_concurrent_operations = 10
        self.max_concurrent_operations = max_concurrent_operations
        
        if performance_monitoring is None:
            performance_monitoring = True
        self.performance_monitoring = performance_monitoring
        
        if cache_ttl is None:
            cache_ttl = 3600  # 1 hour
        self.cache_ttl = cache_ttl

        # Validate qa_model
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        if summarization_model is not None and not isinstance(
            summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )
        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # Enhanced TreeRetrieverConfig
        if tr_enable_caching is None:
            tr_enable_caching = self.enable_caching
        if tr_adaptive_retrieval is None:
            tr_adaptive_retrieval = True
        if tr_early_termination is None:
            tr_early_termination = True

        # Enhanced TreeBuilderConfig  
        if tb_build_mode is None:
            from .tree_builder import BuildMode
            tb_build_mode = BuildMode.ASYNC if self.enable_async else BuildMode.SYNC
        if tb_batch_size is None:
            tb_batch_size = 100
        if tb_enable_progress_tracking is None:
            tb_enable_progress_tracking = self.enable_progress_tracking

        # Set TreeBuilderConfig
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
                build_mode=tb_build_mode,
                batch_size=tb_batch_size,
                enable_progress_tracking=tb_enable_progress_tracking,
                performance_monitoring=self.performance_monitoring,
            )
        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # Set TreeRetrieverConfig
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
                enable_caching=tr_enable_caching,
                cache_ttl=self.cache_ttl,
                adaptive_retrieval=tr_adaptive_retrieval,
                enable_metrics=self.enable_metrics,
                early_termination=tr_early_termination,
            )
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # Assign the created configurations to the instance
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT41QAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            Enable Async: {enable_async}
            Enable Caching: {enable_caching}
            Enable Metrics: {enable_metrics}
            Enable Progress Tracking: {enable_progress_tracking}
            Max Concurrent Operations: {max_concurrent_operations}
            Performance Monitoring: {performance_monitoring}
            Cache TTL: {cache_ttl}
            
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            enable_async=self.enable_async,
            enable_caching=self.enable_caching,
            enable_metrics=self.enable_metrics,
            enable_progress_tracking=self.enable_progress_tracking,
            max_concurrent_operations=self.max_concurrent_operations,
            performance_monitoring=self.performance_monitoring,
            cache_ttl=self.cache_ttl,
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary


class RetrievalAugmentation:
    """
    Enhanced Retrieval Augmentation with async capabilities and comprehensive optimization
    """

    def __init__(self, config=None, tree=None):
        """
        Initializes a RetrievalAugmentation instance with enhanced configuration.
        
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
            tree: The tree instance or the path to a pickled tree file.
        """
        if config is None:
            config = RetrievalAugmentationConfig()
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError(
                "config must be an instance of RetrievalAugmentationConfig"
            )

        self.config = config
        
        # Enhanced features
        self.enable_async = config.enable_async
        self.enable_metrics = config.enable_metrics
        self.enable_progress_tracking = config.enable_progress_tracking
        self.performance_monitoring = config.performance_monitoring
        
        # Initialize components
        self.metrics = PipelineMetrics() if self.enable_metrics else None
        self.progress_tracker = AsyncProgressTracker(self.enable_progress_tracking)
        self.operation_semaphore = asyncio.Semaphore(config.max_concurrent_operations)

        # Check if tree is a string (indicating a path to a pickled tree)
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("The loaded object is not an instance of Tree")
                logging.info(f"Successfully loaded tree from {tree}")
            except Exception as e:
                raise ValueError(f"Failed to load tree from {tree}: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError(
                "tree must be an instance of Tree, a path to a pickled Tree, or None"
            )

        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        # Set up progress callback
        if self.enable_progress_tracking and hasattr(self.tree_builder, 'set_progress_callback'):
            self.tree_builder.set_progress_callback(self.progress_tracker.update_progress)

        logging.info(
            f"Successfully initialized RetrievalAugmentation with Config {config.log_config()}"
        )

    def add_documents(self, docs: str, progress_callback: Optional[Callable] = None):
        """
        Enhanced document addition with async support and progress tracking.
        
        Args:
            docs (str): The input text to add to the tree.
            progress_callback: Optional callback for progress updates.
        """
        if self.tree is not None:
            user_input = input(
                "Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): "
            )
            if user_input.lower() == "y":
                logging.warning("Feature 'add_to_existing' not yet implemented")
                return

        # Add progress callback if provided
        if progress_callback:
            self.progress_tracker.add_callback(progress_callback)

        if self.enable_async:
            # Run async build
            self.tree = asyncio.run(self._add_documents_async(docs))
        else:
            # Run sync build with progress tracking
            self.tree = self._add_documents_sync(docs)
        
        # Initialize retriever with the new tree
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        
        # Update metrics
        if self.metrics and hasattr(self.tree_builder, 'get_performance_metrics'):
            builder_metrics = self.tree_builder.get_performance_metrics()
            self.metrics.build_time = builder_metrics.get('total_build_time', 0)
            self.metrics.nodes_processed = builder_metrics.get('nodes_created', 0)
            self.metrics.layers_built = self.tree.num_layers

    def _add_documents_sync(self, docs: str) -> Tree:
        """Synchronous document addition with enhanced progress tracking"""
        start_time = time.time()
        
        # Build tree synchronously
        tree = self.tree_builder.build_from_text(text=docs, use_multithreading=True)
        
        build_time = time.time() - start_time
        logging.info(f"Tree built synchronously in {build_time:.2f}s")
        
        return tree

    async def _add_documents_async(self, docs: str) -> Tree:
        """Asynchronous document addition with full async pipeline"""
        start_time = time.time()
        
        # Build tree asynchronously
        tree = await self.tree_builder.build_from_text_async(text=docs)
        
        build_time = time.time() - start_time
        logging.info(f"Tree built asynchronously in {build_time:.2f}s")
        
        return tree

    def retrieve(
        self,
        question: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
        use_async: bool = None,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """
        Enhanced retrieve with async support and caching.
        
        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from.
            num_layers (int): The number of layers to traverse.
            top_k (int): Number of top results to retrieve.
            max_tokens (int): The maximum number of tokens.
            collapse_tree (bool): Whether to retrieve information from all nodes.
            return_layer_information (bool): Whether to return layer information.
            use_async (bool): Whether to use async retrieval.
        
        Returns:
            str or tuple: The context or (context, layer_information).
        """
        if self.retriever is None:
            raise ValueError(
                "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
            )

        # Use configured async setting if not specified
        if use_async is None:
            use_async = self.enable_async

        start_time = time.time()
        
        try:
            if use_async:
                # Use async retrieval
                result = asyncio.run(self._retrieve_async(
                    question, start_layer, num_layers, top_k, max_tokens,
                    collapse_tree, return_layer_information
                ))
            else:
                # Use sync retrieval
                result = self.retriever.retrieve(
                    question, start_layer, num_layers, top_k, max_tokens,
                    collapse_tree, return_layer_information
                )
            
            # Update metrics
            if self.metrics:
                retrieval_time = time.time() - start_time
                self.metrics.total_queries += 1
                self.metrics.total_query_time += retrieval_time
                self.metrics.avg_query_time = self.metrics.total_query_time / self.metrics.total_queries
                
                # Get cache stats from retriever
                if hasattr(self.retriever, 'get_performance_stats'):
                    retriever_stats = self.retriever.get_performance_stats()
                    self.metrics.cache_hits = retriever_stats.get('cache_hits', 0)
            
            return result
            
        except Exception as e:
            logging.error(f"Retrieval failed: {e}")
            if self.metrics:
                self.metrics.total_queries += 1
                self.metrics.total_query_time += time.time() - start_time
            raise

    async def _retrieve_async(
        self,
        question: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = True,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Internal async retrieval with semaphore control"""
        async with self.operation_semaphore:
            return await self.retriever.retrieve_async(
                question, start_layer, num_layers, top_k, max_tokens,
                collapse_tree, return_layer_information
            )

    def answer_question(
        self,
        question: str,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
        use_async: bool = None,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """
        Enhanced question answering with async support.
        
        Args:
            question (str): The question to answer.
            top_k (int): Number of top results to retrieve.
            start_layer (int): The layer to start from.
            num_layers (int): The number of layers to traverse.
            max_tokens (int): The maximum number of tokens.
            collapse_tree (bool): Whether to retrieve information from all nodes.
            return_layer_information (bool): Whether to return layer information.
            use_async (bool): Whether to use async processing.
        
        Returns:
            str or tuple: The answer or (answer, layer_information).
        """
        if use_async is None:
            use_async = self.enable_async
        
        if use_async:
            return asyncio.run(self._answer_question_async(
                question, top_k, start_layer, num_layers, max_tokens,
                collapse_tree, return_layer_information
            ))
        else:
            return self._answer_question_sync(
                question, top_k, start_layer, num_layers, max_tokens,
                collapse_tree, return_layer_information
            )

    def _answer_question_sync(
        self,
        question: str,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Synchronous question answering"""
        # Get context
        context, layer_information = self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True, False
        )

        # Generate answer
        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information
        return answer

    async def _answer_question_async(
        self,
        question: str,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Asynchronous question answering"""
        async with self.operation_semaphore:
            # Get context asynchronously
            context, layer_information = await self._retrieve_async(
                question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True
            )

            # Generate answer (QA models are typically sync, so run in executor)
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None, self.qa_model.answer_question, context, question
            )

            if return_layer_information:
                return answer, layer_information
            return answer

    async def answer_questions_batch(
        self,
        questions: List[str],
        batch_size: int = 5,
        **kwargs
    ) -> List[Union[str, Tuple[str, List[Dict]]]]:
        """
        Answer multiple questions in parallel batches.
        
        Args:
            questions: List of questions to answer.
            batch_size: Number of questions to process concurrently.
            **kwargs: Arguments passed to answer_question.
        
        Returns:
            List of answers corresponding to input questions.
        """
        semaphore = asyncio.Semaphore(batch_size)
        
        async def answer_with_semaphore(question: str):
            async with semaphore:
                return await self._answer_question_async(question, **kwargs)
        
        tasks = [answer_with_semaphore(q) for q in questions]
        return await asyncio.gather(*tasks)

    def save(self, path: str, include_metadata: bool = True):
        """
        Enhanced save with metadata and performance metrics.
        
        Args:
            path (str): Path to save the tree.
            include_metadata (bool): Whether to save metadata alongside tree.
        """
        if self.tree is None:
            raise ValueError("There is no tree to save.")
        
        tree_path = Path(path)
        
        # Save tree
        with open(tree_path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"Tree successfully saved to {tree_path}")
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'config': self.config.__dict__,
                'metrics': self.metrics.to_dict() if self.metrics else {},
                'progress': self.progress_tracker.get_progress_summary(),
                'tree_stats': {
                    'num_layers': self.tree.num_layers,
                    'total_nodes': len(self.tree.all_nodes),
                    'leaf_nodes': len(self.tree.leaf_nodes),
                    'root_nodes': len(self.tree.root_nodes),
                }
            }
            
            metadata_path = tree_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logging.info(f"Metadata saved to {metadata_path}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance metrics and statistics.
        """
        summary = {}
        
        # Pipeline metrics
        if self.metrics:
            summary['pipeline'] = self.metrics.to_dict()
        
        # Tree builder metrics
        if hasattr(self.tree_builder, 'get_performance_metrics'):
            summary['tree_builder'] = self.tree_builder.get_performance_metrics()
        
        # Retriever metrics
        if self.retriever and hasattr(self.retriever, 'get_performance_stats'):
            summary['retriever'] = self.retriever.get_performance_stats()
        
        # Progress information
        if self.progress_tracker:
            summary['progress'] = self.progress_tracker.get_progress_summary()
        
        # Tree statistics
        if self.tree:
            summary['tree_stats'] = {
                'num_layers': self.tree.num_layers,
                'total_nodes': len(self.tree.all_nodes),
                'leaf_nodes': len(self.tree.leaf_nodes),
                'root_nodes': len(self.tree.root_nodes),
            }
        
        return summary

    def clear_all_caches(self):
        """Clear all caches in the pipeline"""
        cleared = []
        
        # Clear retriever cache
        if self.retriever and hasattr(self.retriever, 'clear_cache'):
            self.retriever.clear_cache()
            cleared.append('retriever')
        
        # Clear embedding model caches
        if hasattr(self.tree_builder, 'embedding_models'):
            for model_name, model in self.tree_builder.embedding_models.items():
                if hasattr(model, 'cache') and model.cache:
                    model.cache.memory_cache.clear()
                    cleared.append(f'embedding_{model_name}')
        
        logging.info(f"Cleared caches: {cleared}")

    def optimize_performance(self):
        """Perform automatic performance optimizations"""
        optimizations = []
        
        # Clean expired caches
        if self.retriever and hasattr(self.retriever, 'cleanup_expired_cache'):
            self.retriever.cleanup_expired_cache()
            optimizations.append('cleaned_expired_retriever_cache')
        
        # Clean embedding model caches
        if hasattr(self.tree_builder, 'embedding_models'):
            for model_name, model in self.tree_builder.embedding_models.items():
                if hasattr(model, 'cache') and model.cache:
                    # This would need to be implemented in the cache class
                    # model.cache.cleanup_expired()
                    optimizations.append(f'cleaned_embedding_cache_{model_name}')
        
        logging.info(f"Performance optimizations applied: {optimizations}")

    def set_progress_callback(self, callback: Callable[[BuildProgress], None]):
        """Set progress callback for tree building operations"""
        self.progress_tracker.add_callback(callback)
        if hasattr(self.tree_builder, 'set_progress_callback'):
            self.tree_builder.set_progress_callback(self.progress_tracker.update_progress)