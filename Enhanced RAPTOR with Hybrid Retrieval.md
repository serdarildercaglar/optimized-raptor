# Enhanced RAPTOR with Hybrid Retrieval 🚀

## Table of Contents

* [Overview](#overview)
* [New Features](#new-features)
* [Quick Start](#quick-start)
* [Installation](#installation)
* [Usage Guide](#usage-guide)
* [Configuration](#configuration)
* [API Reference](#api-reference)
* [Performance Optimization](#performance-optimization)
* [Evaluation &amp; Benchmarking](#evaluation--benchmarking)
* [Migration Guide](#migration-guide)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)

## Overview

Enhanced RAPTOR extends the original RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) with advanced hybrid retrieval capabilities. This implementation combines  **dense vector search** ,  **sparse keyword matching** ,  **query enhancement** , and **intelligent result fusion** to deliver superior retrieval quality and performance.

### Key Improvements

* 🔄  **Hybrid Retrieval** : Combines dense (semantic) and sparse (keyword) retrieval
* 🧠  **Query Enhancement** : Automatic query expansion, rewriting, and intent detection
* 🎯  **Advanced Fusion** : Multiple algorithms for combining retrieval results
* ⚡  **Performance Optimization** : Async processing, caching, and batch operations
* 📊  **Comprehensive Evaluation** : Built-in metrics and benchmarking tools
* 🔧  **Easy Integration** : Backward compatible with existing RAPTOR code

## New Features

### 1. Sparse Retrieval (BM25)

* **Algorithm Support** : BM25 Okapi, BM25 Plus, TF-IDF
* **Intelligent Caching** : Persistent index caching for fast startup
* **Performance Optimized** : Async support and batch processing

```python
from raptor.sparse_retriever import create_sparse_retriever

# Create BM25 retriever
sparse_retriever = create_sparse_retriever(algorithm="bm25_okapi", k1=1.2, b=0.75)
sparse_retriever.build_from_nodes(tree_nodes)

# Retrieve with scores
results = sparse_retriever.retrieve_with_scores("machine learning", top_k=5)
```

### 2. Query Enhancement

* **Intent Detection** : Factual, definitional, procedural, comparative queries
* **Entity Extraction** : Automatic extraction of key entities
* **Query Expansion** : Semantic and linguistic expansion
* **Query Rewriting** : Multiple query variants for better coverage

```python
from raptor.query_enhancement import create_query_enhancer

# Create enhancer
enhancer = create_query_enhancer(embedding_model, corpus_nodes)

# Enhance query
enhanced = await enhancer.enhance_query("What is artificial intelligence?")
print(f"Intent: {enhanced.intent}")
print(f"Expansions: {enhanced.expanded_terms}")
print(f"Rewrites: {enhanced.rewritten_variants}")
```

### 3. Hybrid Fusion & Reranking

* **Fusion Methods** : Reciprocal Rank Fusion (RRF), Weighted Sum, Borda Count
* **Result Reranking** : Cross-encoder based reranking for improved relevance
* **Confidence Scoring** : Multi-signal confidence estimation

```python
from raptor.hybrid_retriever import HybridRetriever, FusionMethod

# Create hybrid retriever
hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    fusion_method=FusionMethod.RRF,
    dense_weight=0.6,
    sparse_weight=0.4,
    enable_reranking=True
)

# Retrieve with hybrid method
results = await hybrid_retriever.retrieve_hybrid_async("query", top_k=10)
```

### 4. Enhanced RAPTOR Integration

* **Seamless Integration** : Drop-in replacement for standard RAPTOR
* **Multiple Retrieval Modes** : Dense, sparse, or hybrid retrieval
* **Advanced Analytics** : Query analysis and method comparison tools

```python
from raptor.enhanced_retrieval_augmentation import EnhancedRetrievalAugmentation, HybridConfig

# Create enhanced RAPTOR
enhanced_raptor = EnhancedRetrievalAugmentation(
    config=raptor_config,
    hybrid_config=HybridConfig(enable_hybrid=True)
)

# Use different retrieval methods
dense_result = enhanced_raptor.retrieve_enhanced("query", method="dense")
sparse_result = enhanced_raptor.retrieve_enhanced("query", method="sparse") 
hybrid_result = enhanced_raptor.retrieve_enhanced("query", method="hybrid")
```

## Quick Start

### 1. Basic Usage

```python
import os
from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, HybridConfig
)
from raptor.hybrid_retriever import FusionMethod

# Setup models
embed_model = CustomEmbeddingModel()
sum_model = GPT41SummarizationModel()

# Standard RAPTOR config
raptor_config = RetrievalAugmentationConfig(
    tb_max_tokens=100,
    tb_summarization_length=300,
    tb_num_layers=4,
    summarization_model=sum_model,
    embedding_model=embed_model,
    enable_async=True
)

# Hybrid config
hybrid_config = HybridConfig(
    enable_hybrid=True,
    enable_query_enhancement=True,
    enable_sparse_retrieval=True,
    enable_reranking=True,
    fusion_method=FusionMethod.RRF,
    dense_weight=0.6,
    sparse_weight=0.4
)

# Create Enhanced RAPTOR
enhanced_raptor = EnhancedRetrievalAugmentation(
    config=raptor_config,
    hybrid_config=hybrid_config
)

# Build tree from text
text = "Your document text here..."
enhanced_raptor.add_documents(text)

# Use hybrid retrieval
result = enhanced_raptor.retrieve_enhanced(
    "What is the main topic?", 
    method="hybrid",
    top_k=5
)
print(result)
```

### 2. Loading Existing Trees

```python
# Load existing tree with hybrid enhancements
enhanced_raptor = EnhancedRetrievalAugmentation(
    config=raptor_config,
    tree="path/to/existing/tree.pkl",  # Your existing tree
    hybrid_config=hybrid_config
)

# Hybrid components are automatically initialized
```

### 3. Query Analysis

```python
# Analyze query characteristics
analysis = enhanced_raptor.analyze_query("How does machine learning work?")
print(f"Intent: {analysis['enhanced_query']['intent']}")
print(f"Entities: {analysis['enhanced_query']['entities']}")
print(f"Expansions: {analysis['enhanced_query']['expansions']}")
```

### 4. Method Comparison

```python
# Compare different retrieval methods
comparison = enhanced_raptor.compare_retrieval_methods("artificial intelligence")

for method, results in comparison['results'].items():
    print(f"{method}: {results['retrieval_time']:.3f}s, {results['result_count']} results")
```

## Installation

### Basic Installation

```bash
# Install enhanced requirements
pip install -r enhanced_requirements.txt

# Core dependencies
pip install rank-bm25 python-dotenv pathlib
```

### Development Installation

```bash
# Clone repository
git clone <repository-url>
cd enhanced-raptor

# Install with development dependencies
pip install -r enhanced_requirements.txt

# Run tests
pytest test_hybrid_raptor.py -v
```

### Optional Dependencies

```bash
# For advanced NLP preprocessing
pip install spacy nltk

# For additional vector stores
pip install chromadb qdrant-client weaviate-client

# For monitoring and evaluation
pip install wandb tensorboard matplotlib seaborn pandas

# For Redis caching
pip install redis
```

## Usage Guide

### Configuration Options

#### HybridConfig Parameters

```python
hybrid_config = HybridConfig(
    # Core features
    enable_hybrid=True,              # Enable hybrid retrieval
    enable_query_enhancement=True,   # Enable query enhancement
    enable_sparse_retrieval=True,    # Enable BM25 sparse retrieval  
    enable_reranking=True,           # Enable result reranking
  
    # Fusion settings
    fusion_method=FusionMethod.RRF,  # RRF, WEIGHTED_SUM, BORDA_COUNT
    dense_weight=0.6,                # Weight for dense scores
    sparse_weight=0.4,               # Weight for sparse scores
  
    # Sparse retrieval
    sparse_algorithm="bm25_okapi",   # bm25_okapi, bm25_plus, tfidf
    sparse_k1=1.2,                  # BM25 parameter
    sparse_b=0.75,                  # BM25 parameter
  
    # Query enhancement  
    max_query_expansions=5,          # Max expansion terms
    semantic_expansion=True,         # Enable semantic expansion
  
    # Reranking
    rerank_top_k=20,                # Rerank top N results
  
    # Performance
    enable_caching=True,             # Enable caching
    cache_dir="hybrid_cache"         # Cache directory
)
```

#### Retrieval Methods

```python
# Dense retrieval (original RAPTOR)
context = enhanced_raptor.retrieve_enhanced("query", method="dense")

# Sparse retrieval (BM25 only)
context = enhanced_raptor.retrieve_enhanced("query", method="sparse")

# Hybrid retrieval (dense + sparse + fusion + reranking)
context = enhanced_raptor.retrieve_enhanced("query", method="hybrid")

# With detailed results
context, results = enhanced_raptor.retrieve_enhanced(
    "query", 
    method="hybrid", 
    return_detailed=True
)

# Analyze individual results
for result in results:
    print(f"Dense: {result.dense_score:.3f}")
    print(f"Sparse: {result.sparse_score:.3f}")
    print(f"Fused: {result.fused_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
```

### Advanced Features

#### Parameter Optimization

```python
# Optimize hybrid parameters using test queries
test_queries = ["query1", "query2", "query3"]
optimization_results = enhanced_raptor.optimize_hybrid_parameters(test_queries)

print(f"Best weights: Dense={optimization_results['best_parameters']['dense_weight']}")
print(f"Performance score: {optimization_results['best_parameters']['performance_score']}")
```

#### Performance Monitoring

```python
# Get comprehensive performance statistics
performance = enhanced_raptor.get_enhanced_performance_summary()

print("Hybrid Features:")
for feature, enabled in performance['hybrid_features'].items():
    print(f"  {feature}: {'✅' if enabled else '❌'}")

print(f"Cache Hit Rate: {performance['retriever']['cache_hit_rate']:.1%}")
print(f"Avg Query Time: {performance['hybrid_retriever']['avg_retrieval_time']:.3f}s")
```

#### Export/Import Configuration

```python
# Export current configuration
enhanced_raptor.export_hybrid_config("my_hybrid_config.json")

# Configuration includes:
# - All hybrid settings
# - Performance metrics  
# - Optimization results
```

## Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_openai_key
HYBRID_CACHE_DIR=./hybrid_cache
ENABLE_PERFORMANCE_MONITORING=true
DEFAULT_FUSION_METHOD=reciprocal_rank_fusion
DEFAULT_DENSE_WEIGHT=0.6
DEFAULT_SPARSE_WEIGHT=0.4
```

### Configuration Files

```json
// hybrid_config.json
{
  "enable_hybrid": true,
  "enable_query_enhancement": true,
  "fusion_method": "reciprocal_rank_fusion",
  "dense_weight": 0.6,
  "sparse_weight": 0.4,
  "sparse_algorithm": "bm25_okapi",
  "rerank_top_k": 20,
  "enable_caching": true
}
```

### Production Configuration

```python
# Optimized for production
production_config = HybridConfig(
    enable_hybrid=True,
    enable_query_enhancement=True,
    enable_sparse_retrieval=True,
    enable_reranking=True,
  
    # Optimized fusion
    fusion_method=FusionMethod.RRF,
    dense_weight=0.65,  # Slight preference for dense
    sparse_weight=0.35,
  
    # Performance optimization
    enable_caching=True,
    cache_dir="/var/cache/raptor",
    rerank_top_k=15,    # Balance quality vs speed
  
    # Reduced expansion for speed
    max_query_expansions=3,
    semantic_expansion=False  # Disable for faster response
)
```

## API Reference

### EnhancedRetrievalAugmentation

Main class that extends standard RAPTOR with hybrid capabilities.

#### Methods

```python
# Core retrieval
retrieve_enhanced(query, method="hybrid", top_k=10, max_tokens=3500, 
                 enhance_query=True, return_detailed=False)

# Query analysis
enhance_query_only(query) -> EnhancedQuery
analyze_query(query) -> Dict

# Comparison and evaluation
compare_retrieval_methods(query, top_k=5) -> Dict
optimize_hybrid_parameters(test_queries) -> Dict

# Performance and monitoring
get_enhanced_performance_summary() -> Dict
export_hybrid_config(filepath)
```

### HybridConfig

Configuration class for hybrid features.

#### Parameters

```python
HybridConfig(
    enable_hybrid: bool = True,
    enable_query_enhancement: bool = True,
    enable_sparse_retrieval: bool = True,
    enable_reranking: bool = True,
    fusion_method: FusionMethod = FusionMethod.RRF,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    sparse_algorithm: str = "bm25_okapi",
    sparse_k1: float = 1.2,
    sparse_b: float = 0.75,
    max_query_expansions: int = 5,
    semantic_expansion: bool = True,
    rerank_top_k: int = 20,
    enable_caching: bool = True,
    cache_dir: str = "hybrid_cache"
)
```

### Fusion Methods

Available fusion algorithms:

```python
from raptor.hybrid_retriever import FusionMethod

FusionMethod.RRF              # Reciprocal Rank Fusion (recommended)
FusionMethod.WEIGHTED_SUM     # Weighted sum of normalized scores
FusionMethod.BORDA_COUNT      # Borda count voting method
FusionMethod.DISTRIBUTIONAL   # Distribution-based fusion
```

## Performance Optimization

### Caching Strategy

```python
# Enable comprehensive caching
hybrid_config = HybridConfig(
    enable_caching=True,
    cache_dir="./hybrid_cache"  # Persistent cache directory
)

# Cache layers:
# 1. Query result caching (TreeRetriever)
# 2. Embedding caching (EmbeddingModels)  
# 3. Sparse index caching (BM25Retriever)
# 4. Query enhancement caching (QueryEnhancer)
```

### Async Optimization

```python
# Enable async processing for better performance
raptor_config = RetrievalAugmentationConfig(
    enable_async=True,
    max_concurrent_operations=10,
    tb_build_mode="async"
)

# Use async retrieval
result = await enhanced_raptor.retrieve_enhanced("query", method="hybrid")
```

### Batch Processing

```python
# Process multiple queries in batch
queries = ["query1", "query2", "query3", "query4", "query5"]
results = await enhanced_raptor.answer_questions_batch(queries, batch_size=3)
```

### Memory Optimization

```python
# Optimize for large documents
config = RetrievalAugmentationConfig(
    tb_max_tokens=80,          # Smaller chunks
    tb_batch_size=50,          # Smaller batches
    max_concurrent_operations=5 # Reduce concurrency
)

hybrid_config = HybridConfig(
    rerank_top_k=10,           # Rerank fewer results
    max_query_expansions=3,    # Fewer expansions
    semantic_expansion=False   # Disable heavy computation
)
```

## Evaluation & Benchmarking

### Built-in Evaluation

```python
from raptor.evaluation_framework import HybridRAPTOREvaluator, create_sample_evaluation_set

# Create evaluator
evaluator = HybridRAPTOREvaluator(enhanced_raptor, embedding_model)

# Create test queries
eval_queries = create_sample_evaluation_set()

# Run comprehensive evaluation
comparison_df = evaluator.compare_methods(eval_queries)
print(comparison_df)

# Generate detailed report
report_path = evaluator.generate_evaluation_report(
    eval_queries, 
    output_dir="evaluation_results"
)
```

### Custom Evaluation

```python
from raptor.evaluation_framework import EvaluationQuery

# Create custom evaluation queries
custom_queries = [
    EvaluationQuery(
        query="What is machine learning?",
        ground_truth_texts=["ML is a subset of AI...", "Algorithms learn from data..."],
        difficulty="easy",
        category="definitional"
    )
]

# Evaluate
results = evaluator.evaluate_query_set(custom_queries)
```

### Performance Benchmarks

```python
# Run performance benchmarks
pytest test_hybrid_raptor.py -v -m benchmark

# Key metrics measured:
# - Retrieval latency (ms)
# - Throughput (queries/second)  
# - Memory usage
# - Cache efficiency
# - Quality metrics (Precision@K, Recall@K, F1@K, MRR, NDCG)
```

## Migration Guide

### From Standard RAPTOR

Migration is designed to be seamless and backward compatible.

#### Step 1: Update Dependencies

```bash
pip install rank-bm25 python-dotenv
```

#### Step 2: Minimal Changes

```python
# OLD: Standard RAPTOR
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

RA = RetrievalAugmentation(config=config, tree="path/to/tree")
result = RA.retrieve("query")

# NEW: Enhanced RAPTOR (backward compatible!)
from raptor.enhanced_retrieval_augmentation import EnhancedRetrievalAugmentation

# Same interface, enhanced capabilities
enhanced_RA = EnhancedRetrievalAugmentation(config=config, tree="path/to/tree")
result = enhanced_RA.retrieve("query")  # Works exactly the same

# PLUS: New hybrid features
hybrid_result = enhanced_RA.retrieve_enhanced("query", method="hybrid")
```

#### Step 3: Enable Hybrid Features (Optional)

```python
from raptor.enhanced_retrieval_augmentation import HybridConfig

# Add hybrid configuration
hybrid_config = HybridConfig(enable_hybrid=True)

enhanced_RA = EnhancedRetrievalAugmentation(
    config=config, 
    tree="path/to/tree",
    hybrid_config=hybrid_config  # Enable hybrid features
)
```

#### Step 4: Optimize (Optional)

```python
# Test and optimize hybrid parameters
test_queries = ["query1", "query2", "query3"]
optimization = enhanced_RA.optimize_hybrid_parameters(test_queries)

# Use optimized settings
print(f"Recommended dense weight: {optimization['best_parameters']['dense_weight']}")
```

### Migration Checklist

* [ ] Update dependencies (`rank-bm25`, `python-dotenv`)
* [ ] Replace import statement
* [ ] Test existing functionality (should work unchanged)
* [ ] Enable hybrid features with `HybridConfig`
* [ ] Run evaluation to measure improvements
* [ ] Optimize parameters for your use case
* [ ] Update production configuration

## Troubleshooting

### Common Issues

#### 1. Import Errors

```python
# Error: ModuleNotFoundError: No module named 'rank_bm25'
# Solution: 
pip install rank-bm25
```

#### 2. Sparse Retriever Not Initialized

```python
# Error: "Sparse retriever not initialized"
# Solution: Ensure hybrid_config enables sparse retrieval
hybrid_config = HybridConfig(enable_sparse_retrieval=True)
```

#### 3. Query Enhancement Fails

```python
# Error: "Query enhancer not initialized"
# Solution: Enable query enhancement in config
hybrid_config = HybridConfig(enable_query_enhancement=True)
```

#### 4. Slow Performance

```python
# Solution: Optimize configuration for speed
hybrid_config = HybridConfig(
    rerank_top_k=10,           # Reduce reranking load
    max_query_expansions=3,    # Fewer expansions
    semantic_expansion=False,  # Disable heavy computation
    enable_caching=True        # Enable all caching
)
```

#### 5. Memory Issues

```python
# Solution: Reduce memory usage
config = RetrievalAugmentationConfig(
    tb_max_tokens=50,          # Smaller chunks
    tb_batch_size=25,          # Smaller batches
    max_concurrent_operations=3 # Less concurrency
)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging for debugging
enhanced_raptor = EnhancedRetrievalAugmentation(config=config, hybrid_config=hybrid_config)
```

### Performance Diagnostics

```python
# Get detailed performance stats
performance = enhanced_raptor.get_enhanced_performance_summary()

# Check each component
print("Cache Hit Rates:")
for component, stats in performance.items():
    if 'cache_hit_rate' in stats:
        print(f"  {component}: {stats['cache_hit_rate']:.1%}")

print("Query Times:")
for component, stats in performance.items():
    if 'avg_query_time' in stats:
        print(f"  {component}: {stats['avg_query_time']:.3f}s")
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd enhanced-raptor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r enhanced_requirements.txt
pip install -r dev-requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest test_hybrid_raptor.py -v

# Run specific test categories
pytest -m "not slow" -v                    # Skip slow tests
pytest -m "benchmark" -v                   # Only benchmark tests
pytest test_hybrid_raptor.py::TestSparseRetriever -v  # Specific test class
```

### Code Quality

```bash
# Format code
black raptor/
isort raptor/

# Lint code
flake8 raptor/
mypy raptor/

# Check test coverage
pytest --cov=raptor test_hybrid_raptor.py
```

### Adding New Features

1. **Create feature branch** : `git checkout -b feature/new-feature`
2. **Write tests first** : Add tests in `test_hybrid_raptor.py`
3. **Implement feature** : Add code with proper docstrings
4. **Run tests** : Ensure all tests pass
5. **Update documentation** : Update this README and docstrings
6. **Submit PR** : Create pull request with description

### Feature Requests

Priority areas for contributions:

* [ ] Additional vector store integrations (ChromaDB, Qdrant, Weaviate)
* [ ] More sophisticated reranking models
* [ ] Multi-modal support (images, tables, code)
* [ ] Real-time learning and adaptation
* [ ] Advanced evaluation metrics
* [ ] Web UI for configuration and monitoring

---

## License

[Your License Here]

## Citation

If you use Enhanced RAPTOR in your research, please cite:

```bibtex
@software{enhanced_raptor,
  title={Enhanced RAPTOR with Hybrid Retrieval},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

---

**🚀 Ready to supercharge your retrieval system? Get started with Enhanced RAPTOR today!**
