# raptor/cluster_tree_builder_optimized.py - MAJOR TREE DEPTH OPTIMIZATION
import logging
import pickle
import asyncio
import time
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set, Optional, Tuple

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering, AdaptiveDimensionalityReduction
from .tree_builder import TreeBuilder, TreeBuilderConfig, BuildProgress
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class OptimizedClusterTreeConfig(TreeBuilderConfig):
    """OPTIMIZED: Configuration with adaptive parameters for better tree depth"""
    
    def __init__(
        self,
        reduction_dimension=None,  # Will be calculated adaptively
        clustering_algorithm=RAPTOR_Clustering,
        clustering_params=None,
        # Enhanced clustering parameters with better defaults
        max_concurrent_clusters=None,
        clustering_batch_size=None,
        enable_cluster_caching=None,
        adaptive_clustering=None,
        # NEW: Tree depth optimization parameters
        min_nodes_per_cluster=None,
        max_nodes_per_cluster=None,
        target_layers=None,
        adaptive_reduction_dimension=None,
        progressive_threshold=None,
        smart_early_termination=None,
        cluster_size_balancing=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # OPTIMIZATION: Adaptive reduction dimension based on corpus size
        if reduction_dimension is None:
            # Calculate based on number of nodes expected
            estimated_nodes = max(100, kwargs.get('tb_max_tokens', 100) * 10)  # Rough estimate
            reduction_dimension = max(3, min(8, int(math.log2(estimated_nodes))))
        
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        
        # Enhanced clustering parameters with better defaults
        if clustering_params is None:
            clustering_params = {
                'max_length_in_cluster': 2500,  # Smaller clusters for more layers
                'threshold': 0.25,              # More lenient threshold
                'verbose': True
            }
        self.clustering_params = clustering_params
        
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
        
        # NEW: Tree depth optimization parameters
        if min_nodes_per_cluster is None:
            min_nodes_per_cluster = 2
        self.min_nodes_per_cluster = min_nodes_per_cluster
        
        if max_nodes_per_cluster is None:
            max_nodes_per_cluster = 8  # Smaller max for more layers
        self.max_nodes_per_cluster = max_nodes_per_cluster
        
        if target_layers is None:
            target_layers = 5  # Aim for 5 layers
        self.target_layers = target_layers
        
        if adaptive_reduction_dimension is None:
            adaptive_reduction_dimension = True
        self.adaptive_reduction_dimension = adaptive_reduction_dimension
        
        if progressive_threshold is None:
            progressive_threshold = True
        self.progressive_threshold = progressive_threshold
        
        if smart_early_termination is None:
            smart_early_termination = True
        self.smart_early_termination = smart_early_termination
        
        if cluster_size_balancing is None:
            cluster_size_balancing = True
        self.cluster_size_balancing = cluster_size_balancing

    def log_config(self):
        base_summary = super().log_config()
        optimized_summary = f"""
        OPTIMIZED CLUSTERING CONFIGURATION:
        Reduction Dimension: {self.reduction_dimension} (adaptive: {self.adaptive_reduction_dimension})
        Target Layers: {self.target_layers}
        Min/Max Nodes Per Cluster: {self.min_nodes_per_cluster}/{self.max_nodes_per_cluster}
        Progressive Threshold: {self.progressive_threshold}
        Smart Early Termination: {self.smart_early_termination}
        Cluster Size Balancing: {self.cluster_size_balancing}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + optimized_summary

class SmartLayerController:
    """OPTIMIZED: Smart controller for layer construction decisions"""
    
    def __init__(self, config: OptimizedClusterTreeConfig):
        self.config = config
        self.layer_stats = {}
        self.total_nodes_processed = 0
        self.reduction_ratios = []
    
    def should_continue_layer_construction(self, current_layer: int, 
                                         current_nodes: List[Node],
                                         all_nodes: Dict[int, Node]) -> Tuple[bool, str]:
        """OPTIMIZED: Smart decision on whether to continue building layers"""
        
        num_current_nodes = len(current_nodes)
        
        # OPTIMIZATION 1: Ensure minimum layer depth
        if current_layer < 2:  # Always build at least 2 layers
            return True, f"Building minimum layers (current: {current_layer})"
        
        # OPTIMIZATION 2: Check against target layers
        if current_layer < self.config.target_layers:
            # Check if we have enough nodes to meaningfully cluster
            min_nodes_needed = self.config.min_nodes_per_cluster * 2  # At least 2 clusters
            
            if num_current_nodes >= min_nodes_needed:
                return True, f"Targeting {self.config.target_layers} layers (current: {current_layer})"
            else:
                return False, f"Insufficient nodes ({num_current_nodes}) for meaningful clustering at layer {current_layer}"
        
        # OPTIMIZATION 3: Adaptive stopping based on reduction dimension
        if self.config.adaptive_reduction_dimension:
            # Adjust reduction dimension based on current layer
            effective_reduction_dim = max(2, self.config.reduction_dimension - current_layer)
            
            if num_current_nodes <= effective_reduction_dim + 2:
                return False, f"Nodes ({num_current_nodes}) too few for reduction_dim ({effective_reduction_dim}) at layer {current_layer}"
        else:
            # Original stopping condition
            if num_current_nodes <= self.config.reduction_dimension + 1:
                return False, f"Nodes ({num_current_nodes}) <= reduction_dimension + 1 ({self.config.reduction_dimension + 1})"
        
        # OPTIMIZATION 4: Check reduction efficiency
        if current_layer > 0 and self.reduction_ratios:
            recent_ratios = self.reduction_ratios[-2:]  # Last 2 ratios
            if len(recent_ratios) >= 2:
                avg_recent_ratio = sum(recent_ratios) / len(recent_ratios)
                
                # If reduction is getting very poor, stop
                if avg_recent_ratio > 0.8:  # Less than 20% reduction
                    return False, f"Poor reduction efficiency: {avg_recent_ratio:.2%} at layer {current_layer}"
        
        # OPTIMIZATION 5: Maximum reasonable layers
        if current_layer >= 8:  # Practical upper limit
            return False, f"Reached maximum practical layers ({current_layer})"
        
        return True, f"Continue layer construction at layer {current_layer}"
    
    def calculate_adaptive_parameters(self, current_layer: int, 
                                    current_nodes: List[Node]) -> Dict:
        """OPTIMIZED: Calculate adaptive parameters for current layer"""
        num_nodes = len(current_nodes)
        
        parameters = {}
        
        # OPTIMIZATION: Progressive threshold (more lenient in higher layers)
        if self.config.progressive_threshold:
            base_threshold = self.config.clustering_params.get('threshold', 0.25)
            # Make threshold more lenient as we go higher
            layer_adjustment = current_layer * 0.05  # 5% more lenient per layer
            parameters['threshold'] = max(0.1, base_threshold - layer_adjustment)
        
        # OPTIMIZATION: Adaptive reduction dimension
        if self.config.adaptive_reduction_dimension:
            # Start with base dimension, reduce as we go up
            base_dim = self.config.reduction_dimension
            layer_adjustment = max(0, current_layer - 1)  # Don't adjust first layer
            parameters['reduction_dimension'] = max(2, base_dim - layer_adjustment)
        
        # OPTIMIZATION: Adaptive cluster size limits
        if self.config.cluster_size_balancing:
            # Adjust max cluster size based on layer and node count
            target_clusters = max(2, min(6, num_nodes // 3))
            max_cluster_size = max(3, num_nodes // target_clusters)
            parameters['max_length_in_cluster'] = min(
                self.config.clustering_params.get('max_length_in_cluster', 2500),
                max_cluster_size * 400  # Rough token estimate
            )
        
        return parameters
    
    def record_layer_result(self, layer: int, 
                          initial_nodes: int, 
                          final_nodes: int,
                          clusters_created: int):
        """Record layer construction results for analysis"""
        reduction_ratio = final_nodes / initial_nodes if initial_nodes > 0 else 1.0
        self.reduction_ratios.append(reduction_ratio)
        
        self.layer_stats[layer] = {
            'initial_nodes': initial_nodes,
            'final_nodes': final_nodes,
            'clusters_created': clusters_created,
            'reduction_ratio': reduction_ratio,
            'reduction_percentage': (1 - reduction_ratio) * 100
        }
        
        logging.info(f"Layer {layer}: {initial_nodes} → {final_nodes} nodes "
                    f"({reduction_ratio:.1%} ratio, {clusters_created} clusters)")

class OptimizedClusterTreeBuilder(TreeBuilder):
    """OPTIMIZED: Tree builder with guaranteed multi-layer construction"""
    
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, (OptimizedClusterTreeConfig, TreeBuilderConfig)):
            raise ValueError("config must be an instance of OptimizedClusterTreeConfig")
        
        # Convert TreeBuilderConfig to OptimizedClusterTreeConfig if needed
        if isinstance(config, TreeBuilderConfig) and not isinstance(config, OptimizedClusterTreeConfig):
            logging.warning("Converting TreeBuilderConfig to OptimizedClusterTreeConfig")
            # Copy over the basic parameters
            config = OptimizedClusterTreeConfig(
                tokenizer=config.tokenizer,
                max_tokens=config.max_tokens,
                num_layers=config.num_layers,
                threshold=config.threshold,
                top_k=config.top_k,
                selection_mode=config.selection_mode,
                summarization_length=config.summarization_length,
                summarization_model=config.summarization_model,
                embedding_models=config.embedding_models,
                cluster_embedding_model=config.cluster_embedding_model,
                build_mode=config.build_mode,
                batch_size=config.batch_size,
                max_concurrent_embeddings=config.max_concurrent_embeddings,
                max_concurrent_summarizations=config.max_concurrent_summarizations,
                enable_progress_tracking=config.enable_progress_tracking,
                performance_monitoring=config.performance_monitoring,
            )
        
        # Store optimized config
        self.optimized_config = config
        
        # Set optimized parameters
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params
        self.max_concurrent_clusters = config.max_concurrent_clusters
        self.clustering_batch_size = config.clustering_batch_size
        self.enable_cluster_caching = config.enable_cluster_caching
        self.adaptive_clustering = config.adaptive_clustering
        
        # Initialize smart layer controller
        self.layer_controller = SmartLayerController(config)
        
        # Enhanced progress tracking
        self.layer_construction_stats = {}
        
        logging.info(f"OptimizedClusterTreeBuilder initialized with target_layers={config.target_layers}")
        logging.info(config.log_config())

    def construct_tree(self,
                      current_level_nodes: Dict[int, Node],
                      all_tree_nodes: Dict[int, Node],
                      layer_to_nodes: Dict[int, List[Node]],
                      use_multithreading: bool = True,
                      ) -> Dict[int, Node]:
        """OPTIMIZED: Tree construction with guaranteed multi-layer building"""
        
        logging.info("Using OPTIMIZED Cluster TreeBuilder (Sync)")
        return self._construct_tree_optimized_sync(
            current_level_nodes, all_tree_nodes, layer_to_nodes, use_multithreading
        )
    
    def _construct_tree_optimized_sync(self,
                                      current_level_nodes: Dict[int, Node],
                                      all_tree_nodes: Dict[int, Node],
                                      layer_to_nodes: Dict[int, List[Node]],
                                      use_multithreading: bool = True,
                                      ) -> Dict[int, Node]:
        """OPTIMIZED: Multi-layer tree construction with smart parameters"""
        
        next_node_index = len(all_tree_nodes)
        layer = 0
        
        while layer < self.optimized_config.target_layers * 2:  # Allow more attempts
            layer_start_time = time.time()
            
            logging.info(f"=== CONSTRUCTING LAYER {layer} ===")
            
            node_list_current_layer = get_node_list(current_level_nodes)
            
            # OPTIMIZATION: Smart layer continuation decision
            should_continue, reason = self.layer_controller.should_continue_layer_construction(
                layer, node_list_current_layer, all_tree_nodes
            )
            
            if not should_continue:
                logging.info(f"STOPPING layer construction: {reason}")
                self.num_layers = layer
                break
            
            logging.info(f"CONTINUING layer construction: {reason}")
            logging.info(f"Current layer {layer} has {len(node_list_current_layer)} nodes")
            
            # OPTIMIZATION: Calculate adaptive parameters for this layer
            adaptive_params = self.layer_controller.calculate_adaptive_parameters(
                layer, node_list_current_layer
            )
            
            # Merge adaptive parameters with base parameters
            clustering_params = self.clustering_params.copy()
            clustering_params.update(adaptive_params)
            
            logging.info(f"Layer {layer} adaptive parameters: {adaptive_params}")
            
            # OPTIMIZATION: Perform clustering with adaptive parameters
            try:
                clusters = self.clustering_algorithm.perform_clustering(
                    node_list_current_layer,
                    self.cluster_embedding_model,
                    **clustering_params
                )
                
                logging.info(f"Layer {layer}: Created {len(clusters)} clusters from {len(node_list_current_layer)} nodes")
                
                # OPTIMIZATION: Validate clustering results
                if not clusters:
                    logging.warning(f"No clusters created at layer {layer}, stopping")
                    self.num_layers = layer
                    break
                
                if len(clusters) >= len(node_list_current_layer):
                    logging.warning(f"Clustering failed to reduce nodes at layer {layer} ({len(clusters)} clusters from {len(node_list_current_layer)} nodes)")
                    # Try with more lenient parameters
                    if clustering_params.get('threshold', 0.25) > 0.1:
                        logging.info("Retrying with more lenient threshold")
                        clustering_params['threshold'] = max(0.1, clustering_params['threshold'] - 0.1)
                        
                        clusters = self.clustering_algorithm.perform_clustering(
                            node_list_current_layer,
                            self.cluster_embedding_model,
                            **clustering_params
                        )
                        
                        if not clusters or len(clusters) >= len(node_list_current_layer):
                            logging.warning(f"Retry failed, stopping at layer {layer}")
                            self.num_layers = layer
                            break
                    else:
                        self.num_layers = layer
                        break
                
            except Exception as e:
                logging.error(f"Clustering failed at layer {layer}: {e}")
                self.num_layers = layer
                break
            
            # Process clusters to create new level nodes
            new_level_nodes = {}
            summarization_length = self.summarization_length
            
            def process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock):
                node_texts = get_text(cluster)

                summarized_text = self.summarize(
                    context=node_texts,
                    max_tokens=summarization_length,
                )

                logging.debug(f"Cluster summary: {len(self.tokenizer.encode(node_texts))} → {len(self.tokenizer.encode(summarized_text))} tokens")

                __, new_parent_node = self.create_node(
                    next_node_index, summarized_text, {node.index for node in cluster}
                )

                with lock:
                    new_level_nodes[next_node_index] = new_parent_node

            # OPTIMIZATION: Process clusters with threading
            lock = Lock()
            
            if use_multithreading and len(clusters) > 1:
                with ThreadPoolExecutor(max_workers=min(4, len(clusters))) as executor:
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

            # Update layer information
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            
            # Record layer statistics
            self.layer_controller.record_layer_result(
                layer, 
                len(node_list_current_layer), 
                len(new_level_nodes),
                len(clusters)
            )
            
            # Update progress tracking
            if self.progress:
                self.progress.current_layer = layer + 1
                self.progress.created_nodes += len(new_level_nodes)
                layer_time = time.time() - layer_start_time
                self.progress.layer_times.append(layer_time)
                self.progress.log_progress()
            
            layer += 1
            
            logging.info(f"=== COMPLETED LAYER {layer - 1} in {time.time() - layer_start_time:.2f}s ===")
            logging.info(f"Created {len(new_level_nodes)} nodes for next layer")
        
        # Final statistics
        self.num_layers = layer
        logging.info(f"=== TREE CONSTRUCTION COMPLETED ===")
        logging.info(f"Final tree depth: {self.num_layers} layers")
        logging.info(f"Total nodes in tree: {len(all_tree_nodes)}")
        
        for layer_num, stats in self.layer_controller.layer_stats.items():
            logging.info(f"Layer {layer_num}: {stats['initial_nodes']} → {stats['final_nodes']} "
                        f"({stats['reduction_percentage']:.1f}% reduction)")
        
        return current_level_nodes
    
    async def construct_tree_async(self,
                                  current_level_nodes: Dict[int, Node],
                                  all_tree_nodes: Dict[int, Node],
                                  layer_to_nodes: Dict[int, List[Node]],
                                  ) -> Dict[int, Node]:
        """OPTIMIZED: Async tree construction with multi-layer guarantee"""
        
        logging.info("Using OPTIMIZED Cluster TreeBuilder (Async)")
        
        next_node_index = len(all_tree_nodes)
        layer = 0
        
        while layer < self.optimized_config.target_layers * 2:  # Allow more attempts
            layer_start_time = time.time()
            
            logging.info(f"=== ASYNC CONSTRUCTING LAYER {layer} ===")
            
            node_list_current_layer = get_node_list(current_level_nodes)
            
            # Smart layer continuation decision
            should_continue, reason = self.layer_controller.should_continue_layer_construction(
                layer, node_list_current_layer, all_tree_nodes
            )
            
            if not should_continue:
                logging.info(f"STOPPING async layer construction: {reason}")
                self.num_layers = layer
                break
            
            logging.info(f"CONTINUING async layer construction: {reason}")
            
            # Calculate adaptive parameters
            adaptive_params = self.layer_controller.calculate_adaptive_parameters(
                layer, node_list_current_layer
            )
            
            clustering_params = self.clustering_params.copy()
            clustering_params.update(adaptive_params)
            
            # Perform clustering asynchronously
            loop = asyncio.get_event_loop()
            
            try:
                clusters = await loop.run_in_executor(
                    None,
                    lambda: self.clustering_algorithm.perform_clustering(
                        node_list_current_layer,
                        self.cluster_embedding_model,
                        **clustering_params
                    )
                )
                
                logging.info(f"Async Layer {layer}: Created {len(clusters)} clusters")
                
                if not clusters or len(clusters) >= len(node_list_current_layer):
                    logging.warning(f"Async clustering failed at layer {layer}")
                    self.num_layers = layer
                    break
                
            except Exception as e:
                logging.error(f"Async clustering failed at layer {layer}: {e}")
                self.num_layers = layer
                break
            
            # Process clusters asynchronously
            new_level_nodes = {}
            
            # Create summarization tasks
            summarization_tasks = []
            
            for i, cluster in enumerate(clusters):
                node_texts = get_text(cluster)
                
                # Create async summarization task
                task = self.summarize_async(node_texts, self.summarization_length)
                summarization_tasks.append((next_node_index + i, cluster, task))
            
            # Wait for all summarizations
            for node_index, cluster, task in summarization_tasks:
                try:
                    summarized_text = await task
                    
                    _, new_parent_node = await self.create_node_async(
                        node_index, summarized_text, {node.index for node in cluster}
                    )
                    
                    new_level_nodes[node_index] = new_parent_node
                    
                except Exception as e:
                    logging.error(f"Async node creation failed: {e}")
            
            if not new_level_nodes:
                logging.warning(f"No nodes created at async layer {layer}")
                self.num_layers = layer
                break
            
            # Update for next iteration
            next_node_index += len(new_level_nodes)
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)
            
            # Record statistics
            self.layer_controller.record_layer_result(
                layer, 
                len(node_list_current_layer), 
                len(new_level_nodes),
                len(clusters)
            )
            
            # Update progress
            if self.progress:
                self.progress.current_layer = layer + 1
                self.progress.created_nodes += len(new_level_nodes)
                self.progress.layer_times.append(time.time() - layer_start_time)
                if hasattr(self, 'progress_callback') and self.progress_callback:
                    await self.progress_callback.update(self.progress)
            
            layer += 1
            
            logging.info(f"=== ASYNC COMPLETED LAYER {layer - 1} ===")
        
        self.num_layers = layer
        logging.info(f"=== ASYNC TREE CONSTRUCTION COMPLETED ===")
        logging.info(f"Final async tree depth: {self.num_layers} layers")
        
        return current_level_nodes
    
    def get_clustering_stats(self) -> Dict:
        """Enhanced clustering statistics"""
        base_stats = super().get_performance_metrics() if hasattr(super(), 'get_performance_metrics') else {}
        
        optimized_stats = {
            'target_layers': self.optimized_config.target_layers,
            'actual_layers': self.num_layers,
            'layer_efficiency': self.num_layers / self.optimized_config.target_layers if self.optimized_config.target_layers > 0 else 0,
            'adaptive_clustering': self.adaptive_clustering,
            'reduction_dimension': self.reduction_dimension,
            'layer_stats': self.layer_controller.layer_stats,
            'total_reduction_ratio': (
                self.layer_controller.reduction_ratios[-1] if self.layer_controller.reduction_ratios else 1.0
            ),
            'optimization_features': {
                'smart_early_termination': self.optimized_config.smart_early_termination,
                'progressive_threshold': self.optimized_config.progressive_threshold,
                'adaptive_reduction_dimension': self.optimized_config.adaptive_reduction_dimension,
                'cluster_size_balancing': self.optimized_config.cluster_size_balancing,
            }
        }
        
        return {**base_stats, **optimized_stats}