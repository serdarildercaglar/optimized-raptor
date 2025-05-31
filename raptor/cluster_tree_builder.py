import logging
import pickle
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set, Optional, Tuple

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig, BuildProgress
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,
        clustering_params={},
        # New async clustering parameters
        max_concurrent_clusters=None,
        clustering_batch_size=None,
        enable_cluster_caching=None,
        adaptive_clustering=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params
        
        # Enhanced clustering parameters
        if max_concurrent_clusters is None:
            max_concurrent_clusters = 8
        self.max_concurrent_clusters = max_concurrent_clusters
        
        if clustering_batch_size is None:
            clustering_batch_size = 50
        self.clustering_batch_size = clustering_batch_size
        
        if enable_cluster_caching is None:
            enable_cluster_caching = True
        self.enable_cluster_caching = enable_cluster_caching
        
        if adaptive_clustering is None:
            adaptive_clustering = True
        self.adaptive_clustering = adaptive_clustering

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        Max Concurrent Clusters: {self.max_concurrent_clusters}
        Clustering Batch Size: {self.clustering_batch_size}
        Enable Cluster Caching: {self.enable_cluster_caching}
        Adaptive Clustering: {self.adaptive_clustering}
        """
        return base_summary + cluster_tree_summary


class AsyncClusterProcessor:
    """Handles async cluster processing with batching and caching"""
    
    def __init__(self, builder: 'ClusterTreeBuilder'):
        self.builder = builder
        self.cluster_cache = {} if builder.enable_cluster_caching else None
        self.processing_semaphore = asyncio.Semaphore(builder.max_concurrent_clusters)
    
    def _get_cluster_hash(self, nodes: List[Node]) -> str:
        """Generate hash for cluster caching"""
        text_content = "".join(sorted([node.text[:50] for node in nodes]))
        return str(hash(text_content))
    
    async def process_cluster_async(
        self, 
        cluster: List[Node], 
        next_node_index: int,
        summarization_length: int
    ) -> Tuple[int, Node]:
        """Process a single cluster asynchronously"""
        async with self.processing_semaphore:
            start_time = time.time()
            
            # Check cache if enabled
            if self.cluster_cache is not None:
                cluster_hash = self._get_cluster_hash(cluster)
                if cluster_hash in self.cluster_cache:
                    cached_node = self.cluster_cache[cluster_hash]
                    logging.debug(f"Cache hit for cluster {next_node_index}")
                    return next_node_index, cached_node
            
            # Get node texts
            node_texts = get_text(cluster)
            
            # Async summarization
            summarized_text = await self.builder.summarize_async(
                context=node_texts,
                max_tokens=summarization_length,
            )
            
            logging.info(
                f"Cluster {next_node_index}: {len(self.builder.tokenizer.encode(node_texts))} → "
                f"{len(self.builder.tokenizer.encode(summarized_text))} tokens "
                f"({time.time() - start_time:.2f}s)"
            )
            
            # Create new parent node async
            _, new_parent_node = await self.builder.create_node_async(
                next_node_index,
                summarized_text,
                {node.index for node in cluster}
            )
            
            # Cache the result
            if self.cluster_cache is not None:
                self.cluster_cache[cluster_hash] = new_parent_node
            
            return next_node_index, new_parent_node
    
    async def process_clusters_batch(
        self,
        clusters: List[List[Node]],
        next_node_index: int,
        summarization_length: int
    ) -> Dict[int, Node]:
        """Process multiple clusters in parallel"""
        tasks = []
        
        for i, cluster in enumerate(clusters):
            task = self.process_cluster_async(
                cluster, 
                next_node_index + i, 
                summarization_length
            )
            tasks.append(task)
        
        # Process all clusters concurrently
        results = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        new_level_nodes = {}
        for node_index, node in results:
            new_level_nodes[node_index] = node
        
        return new_level_nodes


class AdaptiveClusteringStrategy:
    """Adaptive clustering based on node characteristics"""
    
    @staticmethod
    def should_adjust_parameters(nodes: List[Node], config: ClusterTreeConfig) -> Dict:
        """Determine if clustering parameters should be adjusted"""
        if not config.adaptive_clustering:
            return {}
        
        # Analyze node characteristics
        node_texts = [node.text for node in nodes]
        avg_text_length = sum(len(text) for text in node_texts) / len(node_texts)
        max_text_length = max(len(text) for text in node_texts)
        
        adjustments = {}
        
        # Adjust reduction dimension based on content complexity
        if avg_text_length > 500:  # Long texts
            adjustments['reduction_dimension'] = min(config.reduction_dimension + 2, 15)
        elif avg_text_length < 100:  # Short texts
            adjustments['reduction_dimension'] = max(config.reduction_dimension - 2, 5)
        
        # Adjust clustering parameters based on variance
        if max_text_length / avg_text_length > 3:  # High variance
            adjustments['threshold'] = config.threshold * 0.9  # More lenient
        
        return adjustments


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params
        self.max_concurrent_clusters = config.max_concurrent_clusters
        self.clustering_batch_size = config.clustering_batch_size
        self.enable_cluster_caching = config.enable_cluster_caching
        self.adaptive_clustering = config.adaptive_clustering
        
        # Initialize async components
        self.cluster_processor = AsyncClusterProcessor(self)
        self.adaptive_strategy = AdaptiveClusteringStrategy()

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """Synchronous tree construction (backward compatibility)"""
        logging.info("Using Cluster TreeBuilder (Sync)")
        return self._construct_tree_sync(
            current_level_nodes, all_tree_nodes, layer_to_nodes, use_multithreading
        )
    
    def _construct_tree_sync(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """Original synchronous tree construction with enhancements"""
        next_node_index = len(all_tree_nodes)

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, "
                f"Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):
            layer_start_time = time.time()
            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. "
                    f"Total Layers in tree: {layer}"
                )
                break

            # Adaptive parameter adjustment
            clustering_params = self.clustering_params.copy()
            if self.adaptive_clustering:
                adjustments = self.adaptive_strategy.should_adjust_parameters(
                    node_list_current_layer, self
                )
                clustering_params.update(adjustments)
                if adjustments:
                    logging.info(f"Adaptive clustering adjustments: {adjustments}")

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **clustering_params,
            )

            lock = Lock()
            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)
            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            
            # Update progress
            if self.progress:
                self.progress.current_layer = layer + 1
                self.progress.layer_times.append(time.time() - layer_start_time)
                self.progress.log_progress()

        return new_level_nodes
    
    async def construct_tree_async(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
    ) -> Dict[int, Node]:
        """Fully asynchronous tree construction with parallel cluster processing"""
        logging.info("Using Cluster TreeBuilder (Async)")
        
        next_node_index = len(all_tree_nodes)

        for layer in range(self.num_layers):
            layer_start_time = time.time()
            
            logging.info(f"Constructing Layer {layer} (Async)")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. "
                    f"Total Layers in tree: {layer}"
                )
                break

            # Adaptive parameter adjustment
            clustering_params = self.clustering_params.copy()
            if self.adaptive_clustering:
                adjustments = self.adaptive_strategy.should_adjust_parameters(
                    node_list_current_layer, self
                )
                clustering_params.update(adjustments)
                if adjustments:
                    logging.info(f"Adaptive clustering adjustments: {adjustments}")

            # Perform clustering (this is CPU-bound, so run in executor)
            loop = asyncio.get_event_loop()
            
            # Prepare all arguments for the clustering function
            clustering_args = {
                'max_length_in_cluster': clustering_params.get('max_length_in_cluster', 3500),
                'tokenizer': clustering_params.get('tokenizer', self.tokenizer),
                'reduction_dimension': clustering_params.get('reduction_dimension', self.reduction_dimension),
                'threshold': clustering_params.get('threshold', self.threshold),
                'verbose': clustering_params.get('verbose', False),
            }
            
            # Run clustering in executor with proper argument handling
            clusters = await loop.run_in_executor(
                None,
                lambda: self.clustering_algorithm.perform_clustering(
                    node_list_current_layer,
                    self.cluster_embedding_model,
                    **clustering_args
                )
            )

            logging.info(f"Found {len(clusters)} clusters in layer {layer}")

            # Process clusters in parallel
            new_level_nodes = await self.cluster_processor.process_clusters_batch(
                clusters,
                next_node_index,
                self.summarization_length
            )

            # Update for next iteration
            next_node_index += len(new_level_nodes)
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            
            # Update progress
            if self.progress:
                self.progress.current_layer = layer + 1
                self.progress.created_nodes += len(new_level_nodes)
                self.progress.layer_times.append(time.time() - layer_start_time)
                await self.progress_callback.update(self.progress)
            
            # Update metrics
            if self.metrics:
                self.metrics['summarization_batch_count'] += 1
            
            layer_time = time.time() - layer_start_time
            logging.info(
                f"Layer {layer} completed: {len(new_level_nodes)} nodes "
                f"in {layer_time:.2f}s ({len(new_level_nodes)/layer_time:.1f} nodes/sec)"
            )

        return current_level_nodes
    
    def get_clustering_stats(self) -> Dict:
        """Get clustering-specific performance statistics"""
        stats = self.get_performance_metrics()
        
        # Add clustering-specific metrics
        stats.update({
            'cache_enabled': self.enable_cluster_caching,
            'adaptive_clustering': self.adaptive_clustering,
            'max_concurrent_clusters': self.max_concurrent_clusters,
            'reduction_dimension': self.reduction_dimension,
        })
        
        if hasattr(self.cluster_processor, 'cluster_cache') and self.cluster_processor.cluster_cache:
            stats['cluster_cache_size'] = len(self.cluster_processor.cluster_cache)
        
        return stats