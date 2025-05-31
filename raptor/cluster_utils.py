import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


class ClusteringMethod(Enum):
    """Clustering methods for different cluster sizes"""
    UMAP_GMM = "umap_gmm"           # Large clusters (20+)
    PCA_GMM = "pca_gmm"             # Medium clusters (8-19)
    HIERARCHICAL = "hierarchical"    # Small clusters (3-7)
    DISTANCE_BASED = "distance"      # Very small clusters (2-3)
    NO_CLUSTERING = "none"           # Single nodes


class ClusterQualityMetrics:
    """Quality assessment for clustering results"""
    
    @staticmethod
    def calculate_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics"""
        if len(np.unique(labels)) < 2:
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "inertia": 0.0}
        
        try:
            silhouette = silhouette_score(embeddings, labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, labels)
            
            # Calculate inertia (within-cluster sum of squares)
            inertia = 0.0
            for label in np.unique(labels):
                cluster_points = embeddings[labels == label]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            
            return {
                "silhouette": silhouette,
                "calinski_harabasz": calinski_harabasz,
                "inertia": inertia
            }
        except Exception as e:
            logging.warning(f"Quality metrics calculation failed: {e}")
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "inertia": 0.0}
    
    @staticmethod
    def assess_quality(metrics: Dict[str, float], min_silhouette: float = 0.2) -> bool:
        """Assess if clustering quality meets minimum standards"""
        return metrics.get("silhouette", 0.0) >= min_silhouette


class AdaptiveDimensionalityReduction:
    """Smart dimensionality reduction based on data characteristics"""
    
    @staticmethod
    def select_method(embeddings: np.ndarray, target_dim: int) -> ClusteringMethod:
        """Select optimal dimensionality reduction method based on data size"""
        n_samples = len(embeddings)
        
        if n_samples >= 20:
            return ClusteringMethod.UMAP_GMM
        elif n_samples >= 8:
            return ClusteringMethod.PCA_GMM
        elif n_samples >= 3:
            return ClusteringMethod.HIERARCHICAL
        elif n_samples >= 2:
            return ClusteringMethod.DISTANCE_BASED
        else:
            return ClusteringMethod.NO_CLUSTERING
    
    @staticmethod
    def reduce_dimensions(embeddings: np.ndarray, target_dim: int, method: ClusteringMethod) -> np.ndarray:
        """Apply appropriate dimensionality reduction"""
        if method == ClusteringMethod.UMAP_GMM:
            return AdaptiveDimensionalityReduction._umap_reduction(embeddings, target_dim)
        elif method == ClusteringMethod.PCA_GMM:
            return AdaptiveDimensionalityReduction._pca_reduction(embeddings, target_dim)
        else:
            # For hierarchical and distance-based methods, use original embeddings
            return embeddings
    
    @staticmethod
    def _umap_reduction(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """High-quality UMAP reduction for large clusters"""
        n_samples = len(embeddings)
        
        # Optimal parameters for large clusters
        n_neighbors = min(max(int(np.sqrt(n_samples)), 5), n_samples - 1)
        target_dim = min(target_dim, n_samples - 2)
        
        logging.info(f"UMAP reduction: {n_samples} samples → {target_dim}D (n_neighbors={n_neighbors})")
        
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=target_dim,
                metric='cosine',
                min_dist=0.1,
                spread=1.0,
                random_state=RANDOM_SEED
            )
            return reducer.fit_transform(embeddings)
        except Exception as e:
            logging.warning(f"UMAP failed, falling back to PCA: {e}")
            return AdaptiveDimensionalityReduction._pca_reduction(embeddings, target_dim)
    
    @staticmethod
    def _pca_reduction(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """High-quality PCA reduction for medium clusters"""
        n_samples = len(embeddings)
        target_dim = min(target_dim, n_samples - 1)
        
        logging.info(f"PCA reduction: {n_samples} samples → {target_dim}D")
        
        # Calculate explained variance to ensure we preserve enough information
        pca_full = PCA()
        pca_full.fit(embeddings)
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Ensure we preserve at least 85% of variance
        min_components_for_variance = np.argmax(cumsum_variance >= 0.85) + 1
        target_dim = max(target_dim, min(min_components_for_variance, n_samples - 1))
        
        pca = PCA(n_components=target_dim, random_state=RANDOM_SEED)
        reduced = pca.fit_transform(embeddings)
        
        variance_preserved = np.sum(pca.explained_variance_ratio_)
        logging.info(f"PCA preserved {variance_preserved:.1%} of variance")
        
        return reduced


class AdvancedClustering:
    """Advanced clustering methods with quality assessment"""
    
    @staticmethod
    def cluster_with_method(embeddings: np.ndarray, method: ClusteringMethod, 
                          threshold: float = 0.1, max_clusters: int = None) -> Tuple[List, int, Dict]:
        """Apply clustering method and return results with quality metrics"""
        
        if method == ClusteringMethod.NO_CLUSTERING:
            labels = [np.array([0]) for _ in range(len(embeddings))]
            return labels, 1, {"silhouette": 1.0, "method": "no_clustering"}
        
        elif method == ClusteringMethod.DISTANCE_BASED:
            return AdvancedClustering._distance_based_clustering(embeddings, threshold)
        
        elif method == ClusteringMethod.HIERARCHICAL:
            return AdvancedClustering._hierarchical_clustering(embeddings, max_clusters)
        
        elif method in [ClusteringMethod.UMAP_GMM, ClusteringMethod.PCA_GMM]:
            return AdvancedClustering._gmm_clustering(embeddings, threshold, max_clusters)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    @staticmethod
    def _distance_based_clustering(embeddings: np.ndarray, threshold: float) -> Tuple[List, int, Dict]:
        """Simple distance-based clustering for very small clusters"""
        n_samples = len(embeddings)
        
        if n_samples <= 1:
            labels = [np.array([0]) for _ in range(n_samples)]
            return labels, max(1, n_samples), {"silhouette": 1.0, "method": "distance_based"}
        
        # Calculate pairwise distances
        distances = pdist(embeddings, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Simple threshold-based clustering
        labels = [-1] * n_samples
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] == -1:  # Unassigned
                # Start new cluster
                labels[i] = cluster_id
                
                # Find points within threshold
                for j in range(i + 1, n_samples):
                    if labels[j] == -1 and distance_matrix[i, j] < (1 - threshold):
                        labels[j] = cluster_id
                
                cluster_id += 1
        
        # Convert to expected format
        cluster_labels = [np.array([labels[i]]) for i in range(n_samples)]
        
        # Calculate quality metrics
        label_array = np.array(labels)
        metrics = ClusterQualityMetrics.calculate_metrics(embeddings, label_array)
        metrics["method"] = "distance_based"
        
        logging.info(f"Distance-based clustering: {n_samples} samples → {cluster_id} clusters")
        
        return cluster_labels, cluster_id, metrics
    
    @staticmethod
    def _hierarchical_clustering(embeddings: np.ndarray, max_clusters: int = None) -> Tuple[List, int, Dict]:
        """Hierarchical clustering for small-medium clusters"""
        n_samples = len(embeddings)
        
        if n_samples <= 1:
            labels = [np.array([0]) for _ in range(n_samples)]
            return labels, max(1, n_samples), {"silhouette": 1.0, "method": "hierarchical"}
        
        if max_clusters is None:
            max_clusters = min(n_samples // 2, 5)  # Reasonable default
        
        max_clusters = min(max_clusters, n_samples)
        
        # Try different numbers of clusters and pick the best
        best_labels = None
        best_n_clusters = 1
        best_score = -1
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                cluster_labels = clustering.fit_predict(embeddings)
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    score = silhouette_score(embeddings, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_labels = cluster_labels
                        best_n_clusters = n_clusters
                
            except Exception as e:
                logging.warning(f"Hierarchical clustering failed for n_clusters={n_clusters}: {e}")
                continue
        
        # Fallback to single cluster if all attempts failed
        if best_labels is None:
            best_labels = np.zeros(n_samples, dtype=int)
            best_n_clusters = 1
        
        # Convert to expected format
        cluster_labels = [np.array([best_labels[i]]) for i in range(n_samples)]
        
        # Calculate quality metrics
        metrics = ClusterQualityMetrics.calculate_metrics(embeddings, best_labels)
        metrics["method"] = "hierarchical"
        
        logging.info(f"Hierarchical clustering: {n_samples} samples → {best_n_clusters} clusters (silhouette: {best_score:.3f})")
        
        return cluster_labels, best_n_clusters, metrics
    
    @staticmethod
    def _gmm_clustering(embeddings: np.ndarray, threshold: float, max_clusters: int = None) -> Tuple[List, int, Dict]:
        """GMM clustering for reduced-dimension embeddings"""
        n_samples = len(embeddings)
        
        if max_clusters is None:
            max_clusters = min(n_samples // 2, 10)
        
        max_clusters = min(max_clusters, n_samples)
        
        # Find optimal number of clusters using BIC
        best_n_clusters = 1
        best_bic = float('inf')
        best_gmm = None
        
        for n_clusters in range(1, max_clusters + 1):
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                
                if bic < best_bic:
                    best_bic = bic
                    best_n_clusters = n_clusters
                    best_gmm = gmm
                    
            except Exception as e:
                logging.warning(f"GMM failed for n_components={n_clusters}: {e}")
                continue
        
        if best_gmm is None:
            # Fallback to single cluster
            labels = [np.array([0]) for _ in range(n_samples)]
            return labels, 1, {"silhouette": 1.0, "method": "gmm_fallback"}
        
        # Get cluster probabilities and apply threshold
        probs = best_gmm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        
        # Ensure every point belongs to at least one cluster
        for i, label_set in enumerate(labels):
            if len(label_set) == 0:
                labels[i] = np.array([np.argmax(probs[i])])
        
        # Calculate quality metrics
        hard_labels = np.array([np.argmax(probs[i]) for i in range(n_samples)])
        metrics = ClusterQualityMetrics.calculate_metrics(embeddings, hard_labels)
        metrics["method"] = "gmm"
        
        logging.info(f"GMM clustering: {n_samples} samples → {best_n_clusters} clusters (BIC: {best_bic:.1f})")
        
        return labels, best_n_clusters, metrics


def perform_adaptive_clustering(
    embeddings: np.ndarray, 
    target_dim: int = 10, 
    threshold: float = 0.1, 
    verbose: bool = False,
    quality_threshold: float = 0.2
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Perform adaptive clustering with quality-focused approach
    """
    if len(embeddings) <= 1:
        logging.info("Single or no embeddings - no clustering needed")
        return [np.array([0]) for _ in range(len(embeddings))], {"method": "single", "quality": 1.0}
    
    # Select optimal clustering method
    method = AdaptiveDimensionalityReduction.select_method(embeddings, target_dim)
    
    if verbose:
        logging.info(f"Selected clustering method: {method.value} for {len(embeddings)} embeddings")
    
    # Apply dimensionality reduction if needed
    if method in [ClusteringMethod.UMAP_GMM, ClusteringMethod.PCA_GMM]:
        reduced_embeddings = AdaptiveDimensionalityReduction.reduce_dimensions(
            embeddings, target_dim, method
        )
    else:
        reduced_embeddings = embeddings
    
    # Perform clustering
    max_clusters = min(len(embeddings) // 2, 10)
    cluster_labels, n_clusters, quality_metrics = AdvancedClustering.cluster_with_method(
        reduced_embeddings, method, threshold, max_clusters
    )
    
    # Quality assessment
    is_quality_sufficient = ClusterQualityMetrics.assess_quality(quality_metrics, quality_threshold)
    
    if not is_quality_sufficient and method != ClusteringMethod.NO_CLUSTERING:
        logging.warning(f"Clustering quality insufficient (silhouette: {quality_metrics.get('silhouette', 0):.3f})")
        if len(embeddings) >= 3:
            # Try fallback to hierarchical clustering
            logging.info("Trying hierarchical clustering as fallback")
            cluster_labels, n_clusters, quality_metrics = AdvancedClustering.cluster_with_method(
                embeddings, ClusteringMethod.HIERARCHICAL, threshold, max_clusters
            )
    
    if verbose:
        logging.info(f"Final clustering: {n_clusters} clusters, quality metrics: {quality_metrics}")
    
    # Prepare metadata
    metadata = {
        "method": method.value,
        "n_clusters": n_clusters,
        "quality_metrics": quality_metrics,
        "is_quality_sufficient": is_quality_sufficient,
        "n_embeddings": len(embeddings)
    }
    
    return cluster_labels, metadata


# Legacy functions with enhanced implementation
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Enhanced global clustering with adaptive method selection"""
    method = AdaptiveDimensionalityReduction.select_method(embeddings, dim)
    return AdaptiveDimensionalityReduction.reduce_dimensions(embeddings, dim, method)


def local_cluster_embeddings(
    embeddings: np.ndarray, 
    dim: int, 
    num_neighbors: int = 10, 
    metric: str = "cosine"
) -> np.ndarray:
    """Enhanced local clustering with adaptive method selection"""
    method = AdaptiveDimensionalityReduction.select_method(embeddings, dim)
    return AdaptiveDimensionalityReduction.reduce_dimensions(embeddings, dim, method)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """Enhanced optimal cluster detection"""
    if len(embeddings) <= 2:
        return 1
    
    max_clusters = min(max_clusters, len(embeddings) // 2)
    
    # Use multiple methods and pick the most consistent result
    methods_results = []
    
    # BIC method
    try:
        bic_scores = []
        for n in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            gmm.fit(embeddings)
            bic_scores.append(gmm.bic(embeddings))
        methods_results.append(np.argmin(bic_scores) + 1)
    except:
        pass
    
    # Silhouette method
    try:
        sil_scores = []
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            sil_scores.append(silhouette_score(embeddings, labels))
        methods_results.append(np.argmax(sil_scores) + 2)
    except:
        pass
    
    if methods_results:
        # Return the median result for robustness
        return int(np.median(methods_results))
    else:
        return min(3, len(embeddings) // 2)


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """Enhanced GMM clustering with quality assessment"""
    if len(embeddings) <= 1:
        return [[0]], 1
    
    n_clusters = get_optimal_clusters(embeddings)
    cluster_labels, n_clusters, metrics = AdvancedClustering._gmm_clustering(
        embeddings, threshold, n_clusters
    )
    
    return cluster_labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    """Main clustering function with enhanced quality-focused approach"""
    cluster_labels, metadata = perform_adaptive_clustering(
        embeddings, dim, threshold, verbose
    )
    
    if verbose:
        method = metadata.get("method", "unknown")
        quality = metadata.get("quality_metrics", {}).get("silhouette", 0)
        logging.info(f"Clustering completed: method={method}, quality={quality:.3f}")
    
    return cluster_labels


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    @staticmethod
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("o200k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
        max_recursion_depth: int = 3,
        current_depth: int = 0,
        min_cluster_quality: float = 0.15,
    ) -> List[List[Node]]:
        """
        Enhanced RAPTOR clustering with quality-focused approach and smart recursion control
        """
        if not nodes:
            logging.warning("No nodes provided for clustering")
            return []
        
        if len(nodes) == 1:
            return [nodes]
        
        # Prevent infinite recursion
        if current_depth >= max_recursion_depth:
            logging.info(f"Max recursion depth reached ({current_depth}), stopping recursion")
            return [nodes]
        
        # Skip clustering for very small node sets (but allow some small clustering)
        if len(nodes) <= 2:
            return [nodes]
        
        try:
            # Extract embeddings
            embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
            
            if verbose:
                logging.info(f"Clustering {len(nodes)} nodes at depth {current_depth}")
            
            # Perform adaptive clustering
            clusters, metadata = perform_adaptive_clustering(
                embeddings, 
                reduction_dimension, 
                threshold, 
                verbose,
                min_cluster_quality
            )
            
            # Check clustering quality
            quality_metrics = metadata.get("quality_metrics", {})
            silhouette_score = quality_metrics.get("silhouette", 0)
            
            if verbose:
                logging.info(f"Clustering quality: silhouette={silhouette_score:.3f}, method={metadata.get('method', 'unknown')}")
            
            # Process clusters
            node_clusters = []
            all_labels = set()
            for cluster in clusters:
                all_labels.update(cluster)
            
            if not all_labels:
                return [nodes]
            
            for label in sorted(all_labels):
                # Get nodes for this cluster
                indices = [i for i, cluster in enumerate(clusters) if label in cluster]
                if not indices:
                    continue
                
                cluster_nodes = [nodes[i] for i in indices]
                
                # Smart recursion decision based on content analysis
                should_recurse = RAPTOR_Clustering._should_recurse_cluster(
                    cluster_nodes, tokenizer, max_length_in_cluster, 
                    current_depth, max_recursion_depth, silhouette_score
                )
                
                if should_recurse:
                    if verbose:
                        logging.info(f"Recursively clustering {len(cluster_nodes)} nodes at depth {current_depth}")
                    
                    try:
                        sub_clusters = RAPTOR_Clustering.perform_clustering(
                            cluster_nodes, embedding_model_name, max_length_in_cluster,
                            tokenizer, reduction_dimension, threshold, verbose,
                            max_recursion_depth, current_depth + 1, min_cluster_quality
                        )
                        node_clusters.extend(sub_clusters)
                    except Exception as e:
                        logging.warning(f"Recursive clustering failed: {e}")
                        node_clusters.append(cluster_nodes)
                else:
                    node_clusters.append(cluster_nodes)
            
            return node_clusters if node_clusters else [nodes]
            
        except Exception as e:
            logging.error(f"RAPTOR clustering failed: {e}")
            return [nodes]
    
    @staticmethod
    def _should_recurse_cluster(
        cluster_nodes: List[Node], 
        tokenizer, 
        max_length_in_cluster: int,
        current_depth: int, 
        max_depth: int,
        clustering_quality: float
    ) -> bool:
        """
        Smart decision on whether to recurse based on multiple factors
        """
        # Don't recurse if too few nodes
        if len(cluster_nodes) <= 3:
            return False
        
        # Don't recurse if at max depth
        if current_depth >= max_depth - 1:
            return False
        
        # Calculate cluster size
        total_tokens = sum(len(tokenizer.encode(node.text)) for node in cluster_nodes)
        
        # Don't recurse if cluster is small enough
        if total_tokens <= max_length_in_cluster:
            return False
        
        # Don't recurse if clustering quality was poor (likely no meaningful sub-structure)
        if clustering_quality < 0.1:
            return False
        
        # Recurse if cluster is large and quality suggests sub-structure exists
        return total_tokens > max_length_in_cluster and len(cluster_nodes) >= 4