# raptor/evaluation_framework.py
"""
Evaluation Framework for Enhanced RAPTOR

This module provides comprehensive evaluation capabilities for comparing
retrieval quality, performance, and effectiveness of different methods.
"""

import logging
import asyncio
import time
import json
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class EvaluationQuery:
    """Container for evaluation query and ground truth"""
    query: str
    ground_truth_texts: List[str] = field(default_factory=list)
    ground_truth_answers: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"   # factual, definitional, procedural, etc.
    expected_intent: Optional[str] = None


@dataclass
class RetrievalResult:
    """Container for retrieval evaluation results"""
    query: str
    method: str
    retrieved_texts: List[str]
    scores: List[float]
    retrieval_time: float
    confidence_scores: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Retrieval metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_retrieval_time: float = 0.0
    queries_per_second: float = 0.0
    
    # Quality metrics
    semantic_similarity: float = 0.0
    coverage: float = 0.0
    diversity: float = 0.0
    
    # Method-specific metrics
    method_name: str = ""
    total_queries: int = 0


class RetrievalQualityEvaluator:
    """Evaluate retrieval quality using various metrics"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        
    def calculate_precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0 or not retrieved:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for text in retrieved_k if self._is_relevant(text, relevant))
        
        return relevant_retrieved / min(k, len(retrieved_k))
    
    def calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant or not retrieved:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = sum(1 for text in retrieved_k if self._is_relevant(text, relevant))
        
        return relevant_retrieved / len(relevant)
    
    def calculate_f1_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate F1@K"""
        precision = self.calculate_precision_at_k(retrieved, relevant, k)
        recall = self.calculate_recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_mrr(self, retrieved_list: List[List[str]], relevant_list: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            rank = self._find_first_relevant_rank(retrieved, relevant)
            reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        
        return np.mean(reciprocal_ranks)
    
    def calculate_ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if not retrieved or not relevant:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, text in enumerate(retrieved[:k]):
            relevance = 1.0 if self._is_relevant(text, relevant) else 0.0
            dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = [1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant))
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_semantic_similarity(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate average semantic similarity between retrieved and relevant texts"""
        if not self.embedding_model or not retrieved or not relevant:
            return 0.0
        
        try:
            # Get embeddings
            retrieved_embeddings = [self.embedding_model.create_embedding(text) for text in retrieved]
            relevant_embeddings = [self.embedding_model.create_embedding(text) for text in relevant]
            
            # Calculate pairwise similarities
            similarities = []
            for ret_emb in retrieved_embeddings:
                for rel_emb in relevant_embeddings:
                    similarity = np.dot(ret_emb, rel_emb) / (np.linalg.norm(ret_emb) * np.linalg.norm(rel_emb))
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logging.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_diversity(self, retrieved: List[str]) -> float:
        """Calculate diversity of retrieved results"""
        if len(retrieved) <= 1:
            return 0.0
        
        if not self.embedding_model:
            # Fallback: simple text overlap-based diversity
            return self._calculate_text_diversity(retrieved)
        
        try:
            embeddings = [self.embedding_model.create_embedding(text) for text in retrieved]
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            # Diversity is inverse of average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            return 1.0 - avg_similarity
            
        except Exception as e:
            logging.warning(f"Diversity calculation failed: {e}")
            return self._calculate_text_diversity(retrieved)
    
    def _is_relevant(self, retrieved_text: str, relevant_texts: List[str]) -> bool:
        """Check if retrieved text is relevant to any of the relevant texts"""
        # Simple approach: check for significant overlap
        retrieved_words = set(retrieved_text.lower().split())
        
        for relevant_text in relevant_texts:
            relevant_words = set(relevant_text.lower().split())
            overlap = len(retrieved_words.intersection(relevant_words))
            
            # Consider relevant if >30% overlap or exact match
            if overlap / max(len(retrieved_words), 1) > 0.3 or retrieved_text.strip() in relevant_text:
                return True
        
        return False
    
    def _find_first_relevant_rank(self, retrieved: List[str], relevant: List[str]) -> int:
        """Find rank of first relevant document (1-indexed, 0 if none found)"""
        for i, text in enumerate(retrieved):
            if self._is_relevant(text, relevant):
                return i + 1
        return 0
    
    def _calculate_text_diversity(self, texts: List[str]) -> float:
        """Calculate text diversity using word overlap"""
        if len(texts) <= 1:
            return 0.0
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                words_i = set(texts[i].lower().split())
                words_j = set(texts[j].lower().split())
                
                if len(words_i) == 0 or len(words_j) == 0:
                    similarity = 0.0
                else:
                    similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity


class PerformanceEvaluator:
    """Evaluate retrieval performance metrics"""
    
    def __init__(self):
        self.measurements = defaultdict(list)
    
    def measure_retrieval_time(self, retrieval_func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure retrieval time and return result + time"""
        start_time = time.time()
        result = retrieval_func(*args, **kwargs)
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        self.measurements['retrieval_times'].append(retrieval_time)
        
        return result, retrieval_time
    
    async def measure_async_retrieval_time(self, async_retrieval_func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure async retrieval time"""
        start_time = time.time()
        result = await async_retrieval_func(*args, **kwargs)
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        self.measurements['async_retrieval_times'].append(retrieval_time)
        
        return result, retrieval_time
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics"""
        summary = {}
        
        for metric_name, measurements in self.measurements.items():
            if measurements:
                summary[f"{metric_name}_mean"] = np.mean(measurements)
                summary[f"{metric_name}_std"] = np.std(measurements)
                summary[f"{metric_name}_min"] = np.min(measurements)
                summary[f"{metric_name}_max"] = np.max(measurements)
                summary[f"{metric_name}_p95"] = np.percentile(measurements, 95)
        
        return summary


class HybridRAPTOREvaluator:
    """Main evaluator for Enhanced RAPTOR with hybrid features"""
    
    def __init__(self, enhanced_raptor, embedding_model=None):
        self.enhanced_raptor = enhanced_raptor
        self.quality_evaluator = RetrievalQualityEvaluator(embedding_model)
        self.performance_evaluator = PerformanceEvaluator()
        
    def evaluate_single_query(self, eval_query: EvaluationQuery, 
                            methods: List[str] = None) -> Dict[str, EvaluationMetrics]:
        """Evaluate a single query across different methods"""
        if methods is None:
            methods = ["dense", "sparse", "hybrid"]
        
        results = {}
        
        for method in methods:
            try:
                # Measure retrieval
                start_time = time.time()
                
                if method == "hybrid" and hasattr(self.enhanced_raptor, 'retrieve_enhanced'):
                    context, detailed_results = self.enhanced_raptor.retrieve_enhanced(
                        eval_query.query, method=method, return_detailed=True, top_k=10
                    )
                    retrieved_texts = [result.node.text for result in detailed_results] if detailed_results else [context]
                    scores = [result.fused_score for result in detailed_results] if detailed_results else [1.0]
                    confidence_scores = [getattr(result, 'confidence', 0.0) for result in detailed_results] if detailed_results else [0.0]
                else:
                    # Fallback to standard retrieval
                    context = self.enhanced_raptor.retrieve(eval_query.query, top_k=10)
                    retrieved_texts = [context]
                    scores = [1.0]
                    confidence_scores = [0.0]
                
                retrieval_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_metrics_for_method(
                    eval_query, retrieved_texts, scores, retrieval_time, method, confidence_scores
                )
                
                results[method] = metrics
                
            except Exception as e:
                logging.warning(f"Evaluation failed for method {method}: {e}")
                # Create empty metrics for failed method
                results[method] = EvaluationMetrics(method_name=method, total_queries=1)
        
        return results
    
    def evaluate_query_set(self, eval_queries: List[EvaluationQuery], 
                          methods: List[str] = None) -> Dict[str, EvaluationMetrics]:
        """Evaluate a set of queries and return aggregated metrics"""
        if methods is None:
            methods = ["dense", "sparse", "hybrid"]
        
        # Initialize aggregated metrics
        aggregated_metrics = {method: EvaluationMetrics(method_name=method) for method in methods}
        
        for eval_query in eval_queries:
            query_results = self.evaluate_single_query(eval_query, methods)
            
            # Aggregate metrics
            for method, metrics in query_results.items():
                if method in aggregated_metrics:
                    self._aggregate_metrics(aggregated_metrics[method], metrics)
        
        # Finalize aggregated metrics
        for method, metrics in aggregated_metrics.items():
            if metrics.total_queries > 0:
                self._finalize_metrics(metrics)
        
        return aggregated_metrics
    
    def compare_methods(self, eval_queries: List[EvaluationQuery], 
                       methods: List[str] = None) -> pd.DataFrame:
        """Compare different methods and return comparison DataFrame"""
        metrics = self.evaluate_query_set(eval_queries, methods)
        
        # Create comparison data
        comparison_data = []
        
        for method, metric in metrics.items():
            row = {
                'Method': method,
                'Precision@5': metric.precision_at_k.get(5, 0.0),
                'Recall@5': metric.recall_at_k.get(5, 0.0),
                'F1@5': metric.f1_at_k.get(5, 0.0),
                'MRR': metric.mrr,
                'NDCG@5': metric.ndcg_at_k.get(5, 0.0),
                'Avg_Retrieval_Time': metric.avg_retrieval_time,
                'Semantic_Similarity': metric.semantic_similarity,
                'Diversity': metric.diversity,
                'Total_Queries': metric.total_queries
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_evaluation_report(self, eval_queries: List[EvaluationQuery], 
                                 output_dir: str = "evaluation_results") -> str:
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logging.info(f"Generating evaluation report for {len(eval_queries)} queries...")
        
        # Run evaluation
        metrics = self.evaluate_query_set(eval_queries)
        comparison_df = self.compare_methods(eval_queries)
        
        # Save results
        comparison_df.to_csv(output_path / "method_comparison.csv", index=False)
        
        # Generate visualizations
        self._create_visualizations(comparison_df, output_path)
        
        # Generate text report
        report_path = output_path / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(self._generate_text_report(comparison_df, metrics))
        
        # Save detailed metrics
        with open(output_path / "detailed_metrics.json", 'w') as f:
            json.dump({method: self._metrics_to_dict(metric) for method, metric in metrics.items()}, 
                     f, indent=2, default=str)
        
        logging.info(f"Evaluation report saved to {output_path}")
        return str(report_path)
    
    def _calculate_metrics_for_method(self, eval_query: EvaluationQuery, 
                                    retrieved_texts: List[str], scores: List[float],
                                    retrieval_time: float, method: str,
                                    confidence_scores: List[float]) -> EvaluationMetrics:
        """Calculate metrics for a single method and query"""
        metrics = EvaluationMetrics(method_name=method, total_queries=1)
        
        # Retrieval quality metrics
        for k in [1, 3, 5, 10]:
            metrics.precision_at_k[k] = self.quality_evaluator.calculate_precision_at_k(
                retrieved_texts, eval_query.ground_truth_texts, k
            )
            metrics.recall_at_k[k] = self.quality_evaluator.calculate_recall_at_k(
                retrieved_texts, eval_query.ground_truth_texts, k
            )
            metrics.f1_at_k[k] = self.quality_evaluator.calculate_f1_at_k(
                retrieved_texts, eval_query.ground_truth_texts, k
            )
            metrics.ndcg_at_k[k] = self.quality_evaluator.calculate_ndcg_at_k(
                retrieved_texts, eval_query.ground_truth_texts, k
            )
        
        # MRR for single query
        metrics.mrr = self.quality_evaluator.calculate_mrr(
            [retrieved_texts], [eval_query.ground_truth_texts]
        )
        
        # Performance metrics
        metrics.avg_retrieval_time = retrieval_time
        metrics.queries_per_second = 1.0 / retrieval_time if retrieval_time > 0 else 0.0
        
        # Quality metrics
        metrics.semantic_similarity = self.quality_evaluator.calculate_semantic_similarity(
            retrieved_texts, eval_query.ground_truth_texts
        )
        metrics.diversity = self.quality_evaluator.calculate_diversity(retrieved_texts)
        
        # Coverage (what fraction of ground truth is covered)
        if eval_query.ground_truth_texts:
            covered = sum(1 for gt in eval_query.ground_truth_texts 
                         if any(self.quality_evaluator._is_relevant(rt, [gt]) for rt in retrieved_texts))
            metrics.coverage = covered / len(eval_query.ground_truth_texts)
        
        return metrics
    
    def _aggregate_metrics(self, aggregated: EvaluationMetrics, new_metrics: EvaluationMetrics):
        """Aggregate metrics from multiple queries"""
        n = aggregated.total_queries
        
        # Update counts
        aggregated.total_queries += 1
        
        # Update averages
        for k in new_metrics.precision_at_k:
            if k not in aggregated.precision_at_k:
                aggregated.precision_at_k[k] = 0.0
            aggregated.precision_at_k[k] = (aggregated.precision_at_k[k] * n + new_metrics.precision_at_k[k]) / (n + 1)
        
        for k in new_metrics.recall_at_k:
            if k not in aggregated.recall_at_k:
                aggregated.recall_at_k[k] = 0.0
            aggregated.recall_at_k[k] = (aggregated.recall_at_k[k] * n + new_metrics.recall_at_k[k]) / (n + 1)
        
        for k in new_metrics.f1_at_k:
            if k not in aggregated.f1_at_k:
                aggregated.f1_at_k[k] = 0.0
            aggregated.f1_at_k[k] = (aggregated.f1_at_k[k] * n + new_metrics.f1_at_k[k]) / (n + 1)
        
        for k in new_metrics.ndcg_at_k:
            if k not in aggregated.ndcg_at_k:
                aggregated.ndcg_at_k[k] = 0.0
            aggregated.ndcg_at_k[k] = (aggregated.ndcg_at_k[k] * n + new_metrics.ndcg_at_k[k]) / (n + 1)
        
        # Update other metrics
        aggregated.mrr = (aggregated.mrr * n + new_metrics.mrr) / (n + 1)
        aggregated.avg_retrieval_time = (aggregated.avg_retrieval_time * n + new_metrics.avg_retrieval_time) / (n + 1)
        aggregated.semantic_similarity = (aggregated.semantic_similarity * n + new_metrics.semantic_similarity) / (n + 1)
        aggregated.diversity = (aggregated.diversity * n + new_metrics.diversity) / (n + 1)
        aggregated.coverage = (aggregated.coverage * n + new_metrics.coverage) / (n + 1)
    
    def _finalize_metrics(self, metrics: EvaluationMetrics):
        """Finalize aggregated metrics"""
        if metrics.avg_retrieval_time > 0:
            metrics.queries_per_second = 1.0 / metrics.avg_retrieval_time
    
    def _create_visualizations(self, comparison_df: pd.DataFrame, output_path: Path):
        """Create visualization plots"""
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Precision/Recall/F1 comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Precision@5
            axes[0, 0].bar(comparison_df['Method'], comparison_df['Precision@5'])
            axes[0, 0].set_title('Precision@5 by Method')
            axes[0, 0].set_ylabel('Precision@5')
            
            # Recall@5
            axes[0, 1].bar(comparison_df['Method'], comparison_df['Recall@5'])
            axes[0, 1].set_title('Recall@5 by Method')
            axes[0, 1].set_ylabel('Recall@5')
            
            # F1@5
            axes[1, 0].bar(comparison_df['Method'], comparison_df['F1@5'])
            axes[1, 0].set_title('F1@5 by Method')
            axes[1, 0].set_ylabel('F1@5')
            
            # Retrieval Time
            axes[1, 1].bar(comparison_df['Method'], comparison_df['Avg_Retrieval_Time'])
            axes[1, 1].set_title('Average Retrieval Time by Method')
            axes[1, 1].set_ylabel('Time (seconds)')
            
            plt.tight_layout()
            plt.savefig(output_path / "method_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Radar chart for overall comparison
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            metrics_to_plot = ['Precision@5', 'Recall@5', 'F1@5', 'MRR', 'NDCG@5', 'Semantic_Similarity']
            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False)
            
            for idx, row in comparison_df.iterrows():
                values = [row[metric] for metric in metrics_to_plot]
                values += values[:1]  # Complete the circle
                angles_plot = np.concatenate([angles, [angles[0]]])
                
                ax.plot(angles_plot, values, 'o-', linewidth=2, label=row['Method'])
                ax.fill(angles_plot, values, alpha=0.25)
            
            ax.set_xticks(angles)
            ax.set_xticklabels(metrics_to_plot)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            ax.set_title("Overall Method Comparison", size=16, y=1.1)
            
            plt.savefig(output_path / "radar_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.warning(f"Visualization creation failed: {e}")
    
    def _generate_text_report(self, comparison_df: pd.DataFrame, 
                            detailed_metrics: Dict[str, EvaluationMetrics]) -> str:
        """Generate text-based evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED RAPTOR EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        best_method = comparison_df.loc[comparison_df['F1@5'].idxmax(), 'Method']
        best_f1 = comparison_df['F1@5'].max()
        
        report.append(f"• Best performing method: {best_method} (F1@5: {best_f1:.3f})")
        report.append(f"• Total queries evaluated: {comparison_df['Total_Queries'].iloc[0]}")
        report.append("")
        
        # Method comparison
        report.append("METHOD COMPARISON")
        report.append("-" * 40)
        
        for idx, row in comparison_df.iterrows():
            report.append(f"\n{row['Method'].upper()}:")
            report.append(f"  Precision@5: {row['Precision@5']:.3f}")
            report.append(f"  Recall@5: {row['Recall@5']:.3f}")
            report.append(f"  F1@5: {row['F1@5']:.3f}")
            report.append(f"  MRR: {row['MRR']:.3f}")
            report.append(f"  NDCG@5: {row['NDCG@5']:.3f}")
            report.append(f"  Avg Retrieval Time: {row['Avg_Retrieval_Time']:.3f}s")
            report.append(f"  Semantic Similarity: {row['Semantic_Similarity']:.3f}")
            report.append(f"  Diversity: {row['Diversity']:.3f}")
        
        # Performance analysis
        report.append("\n\nPERFORMANCE ANALYSIS")
        report.append("-" * 40)
        
        fastest_method = comparison_df.loc[comparison_df['Avg_Retrieval_Time'].idxmin(), 'Method']
        fastest_time = comparison_df['Avg_Retrieval_Time'].min()
        
        report.append(f"• Fastest method: {fastest_method} ({fastest_time:.3f}s)")
        
        if 'hybrid' in comparison_df['Method'].values:
            hybrid_row = comparison_df[comparison_df['Method'] == 'hybrid'].iloc[0]
            dense_row = comparison_df[comparison_df['Method'] == 'dense'].iloc[0] if 'dense' in comparison_df['Method'].values else None
            
            if dense_row is not None:
                f1_improvement = ((hybrid_row['F1@5'] - dense_row['F1@5']) / dense_row['F1@5']) * 100
                report.append(f"• Hybrid vs Dense F1@5 improvement: {f1_improvement:.1f}%")
        
        # Recommendations
        report.append("\n\nRECOMMENDATIONS")
        report.append("-" * 40)
        
        if best_method == "hybrid":
            report.append("• ✅ Hybrid retrieval shows best overall performance")
            report.append("• Consider using hybrid method for production")
        else:
            report.append(f"• Consider optimizing hybrid parameters")
            report.append(f"• {best_method} method currently performs best")
        
        if comparison_df['Avg_Retrieval_Time'].max() > 1.0:
            report.append("• ⚠️ Consider optimizing retrieval speed for better user experience")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _metrics_to_dict(self, metrics: EvaluationMetrics) -> Dict:
        """Convert EvaluationMetrics to dictionary"""
        return {
            'precision_at_k': metrics.precision_at_k,
            'recall_at_k': metrics.recall_at_k,
            'f1_at_k': metrics.f1_at_k,
            'mrr': metrics.mrr,
            'ndcg_at_k': metrics.ndcg_at_k,
            'avg_retrieval_time': metrics.avg_retrieval_time,
            'queries_per_second': metrics.queries_per_second,
            'semantic_similarity': metrics.semantic_similarity,
            'coverage': metrics.coverage,
            'diversity': metrics.diversity,
            'method_name': metrics.method_name,
            'total_queries': metrics.total_queries
        }


# Sample evaluation queries for testing
def create_sample_evaluation_set() -> List[EvaluationQuery]:
    """Create sample evaluation queries for testing"""
    
    return [
        EvaluationQuery(
            query="What is artificial intelligence?",
            ground_truth_texts=[
                "Artificial intelligence is a branch of computer science",
                "AI aims to create intelligent machines",
                "Machine intelligence and computer science"
            ],
            difficulty="easy",
            category="definitional"
        ),
        EvaluationQuery(
            query="How does machine learning work?",
            ground_truth_texts=[
                "Machine learning enables computers to learn from data",
                "Algorithms learn patterns without explicit programming",
                "Training data is used to build models"
            ],
            difficulty="medium",
            category="procedural"
        ),
        EvaluationQuery(
            query="Compare deep learning and traditional machine learning",
            ground_truth_texts=[
                "Deep learning uses neural networks with multiple layers",
                "Traditional ML uses simpler algorithms",
                "Deep learning requires more data and computation"
            ],
            difficulty="hard",
            category="comparative"
        ),
        EvaluationQuery(
            query="AI ethics challenges",
            ground_truth_texts=[
                "Bias in AI systems",
                "Privacy concerns with data collection",
                "Job displacement due to automation",
                "Algorithmic transparency and explainability"
            ],
            difficulty="medium",
            category="analytical"
        ),
        EvaluationQuery(
            query="natural language processing applications",
            ground_truth_texts=[
                "NLP enables text analysis and generation",
                "Applications include translation and summarization",
                "Chatbots and virtual assistants use NLP"
            ],
            difficulty="easy",
            category="factual"
        )
    ]


# Example usage
def run_evaluation_example():
    """Example of how to run evaluation"""
    
    # This would be your actual enhanced RAPTOR instance
    # enhanced_raptor = EnhancedRetrievalAugmentation(...)
    
    print("Enhanced RAPTOR Evaluation Example")
    print("=" * 50)
    
    print("""
    # 1. Create evaluator
    evaluator = HybridRAPTOREvaluator(enhanced_raptor, embedding_model)
    
    # 2. Create evaluation queries
    eval_queries = create_sample_evaluation_set()
    
    # 3. Run evaluation
    comparison_df = evaluator.compare_methods(eval_queries)
    print(comparison_df)
    
    # 4. Generate comprehensive report
    report_path = evaluator.generate_evaluation_report(
        eval_queries, 
        output_dir="evaluation_results"
    )
    
    # 5. Print results
    print(f"Evaluation report saved to: {report_path}")
    """)


if __name__ == "__main__":
    run_evaluation_example()