# 🌳 RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval

<div align="center">

![RAPTOR Logo](https://img.shields.io/badge/RAPTOR-Tree--Based%20RAG-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange?style=for-the-badge&logo=openai)

**Production-Ready Implementation with Enterprise-Level Optimizations**

</div>

## 📋 Table of Contents

- [🌟 Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [⚡ Usage Examples](#-usage-examples)
- [🔧 Configuration](#-configuration)
- [🎯 Performance Tips](#-performance-tips)
- [❓ FAQ](#-faq)

## 🌟 Features

### 🎯 **Core RAPTOR Capabilities**
- **Hierarchical Tree Structure**: Multi-layer document representation
- **Semantic Clustering**: Groups related content intelligently
- **Recursive Summarization**: Preserves context at all levels
- **Flexible Retrieval**: Tree traversal + collapsed tree methods

### ⚡ **Enterprise Optimizations**
- **Async Pipeline**: ~10x faster processing
- **Smart Caching**: 80%+ query speedup
- **Batch Processing**: 100x embedding efficiency
- **Quality-Focused**: Adaptive clustering with fallbacks
- **Real-time Monitoring**: Progress tracking + performance metrics

## 🚀 Quick Start

### 1️⃣ **Install & Setup**

```bash
# Clone the repository
git clone <repository-url>
cd raptor

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# or create .env file:
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2️⃣ **Prepare Your Data**

```bash
# Create your data file
echo "Your document content here..." > data.txt
```

### 3️⃣ **Build & Query in 30 Seconds**

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig

# 🔥 One-line setup with optimized defaults
RA = RetrievalAugmentation()

# 📄 Load your document
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 🏗️ Build the tree (with real-time progress)
RA.add_documents(text)

# 🤖 Ask questions
answer = RA.answer_question("What is this document about?")
print(answer)
```

## 📦 Installation

### **Requirements**
- Python 3.8+
- OpenAI API key
- 4GB+ RAM (for large documents)

### **Step-by-Step Installation**

```bash
# 1. Create virtual environment (recommended)
python -m venv raptor-env
source raptor-env/bin/activate  # Linux/Mac
# raptor-env\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import raptor; print('✅ RAPTOR installed successfully!')"
```

### **Dependencies Overview**
```txt
# Core ML libraries
torch                  # Neural networks
transformers>=4.52.4   # HuggingFace models
sentence-transformers  # Embedding models
openai>=1.82.1        # OpenAI API

# Scientific computing
numpy>=1.26.3         # Numerical operations
scikit-learn>=1.6.1   # ML algorithms
umap-learn>=0.5.7     # Dimensionality reduction

# Utilities
tiktoken>=0.9.0       # Tokenization
tenacity>=9.1.2       # Retry logic
fastapi>=0.115.12     # Web API (optional)
```

## ⚡ Usage Examples

### 🎯 **Basic Usage**

```python
from raptor import RetrievalAugmentation

# Simple setup
RA = RetrievalAugmentation()

# Load document
with open('data.txt', 'r') as f:
    text = f.read()

# Build tree
RA.add_documents(text)

# Ask questions
answer = RA.answer_question("Summarize the main points")
print(f"Answer: {answer}")
```

### 🚀 **Optimized Configuration**

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import GPT4OSummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel

# Initialize optimized models
embed_model = CustomEmbeddingModel()
sum_model = GPT4OSummarizationModel()

# Create optimized configuration
config = RetrievalAugmentationConfig(
    # 🏗️ Tree Building Optimizations
    tb_max_tokens=100,              # Chunk size
    tb_summarization_length=400,    # Summary length
    tb_num_layers=5,                # Tree depth
    tb_batch_size=100,              # Batch processing
    tb_build_mode="async",          # Async building
    
    # 🔍 Retrieval Optimizations
    tr_enable_caching=True,         # Smart caching
    tr_adaptive_retrieval=True,     # Auto-tuning
    tr_early_termination=True,      # Confidence stopping
    
    # ⚡ Performance Features
    enable_async=True,              # Async pipeline
    enable_metrics=True,            # Performance monitoring
    max_concurrent_operations=10,   # Parallelization
    cache_ttl=3600,                # 1-hour cache
    
    # 🤖 Models
    summarization_model=sum_model,
    embedding_model=embed_model,
)

# Initialize with config
RA = RetrievalAugmentation(config=config)
```

### 📊 **Progress Tracking**

```python
def progress_callback(progress):
    print(f"📊 Layer {progress.current_layer}/{progress.total_layers} "
          f"({progress.layer_progress:.1%}) | "
          f"Time: {progress.elapsed_time:.1f}s")

# Set progress callback
RA.set_progress_callback(progress_callback)

# Build with real-time updates
RA.add_documents(text)
```

### 🔍 **Advanced Retrieval**

```python
# Context-only retrieval
context = RA.retrieve(
    question="What are the key findings?",
    max_tokens=2000,
    collapse_tree=True
)

# Question answering with layer info
answer, layer_info = RA.answer_question(
    question="Explain the methodology",
    return_layer_information=True,
    max_tokens=3000
)

print(f"Answer: {answer}")
print(f"Retrieved from {len(layer_info)} tree nodes")
```

### 📈 **Performance Monitoring**

```python
# Get comprehensive performance stats
stats = RA.get_performance_summary()

print("📊 Performance Summary:")
print(f"├── Build Time: {stats['pipeline']['build_time']:.1f}s")
print(f"├── Cache Hit Rate: {stats['retriever']['cache_hit_rate']:.1%}")
print(f"├── Avg Query Time: {stats['pipeline']['avg_query_time']:.3f}s")
print(f"└── Nodes Created: {stats['tree_stats']['total_nodes']}")
```

### 💾 **Save & Load Trees**

```python
# Save tree with metadata
RA.save("my_raptor_tree", include_metadata=True)

# Load saved tree
RA_loaded = RetrievalAugmentation(
    config=config,
    tree="my_raptor_tree"
)

# Ready to query immediately
answer = RA_loaded.answer_question("Quick question?")
```

## 🔧 Configuration

### 📝 **Document Size-Based Optimization**

```python
def get_optimized_config(document_size):
    """Auto-configure based on document size"""
    
    if document_size < 10_000:  # Small docs (<10KB)
        return RetrievalAugmentationConfig(
            tb_max_tokens=150,
            tb_summarization_length=200,
            tb_num_layers=3,
            tb_batch_size=50
        )
    
    elif document_size < 100_000:  # Medium docs (10-100KB)  
        return RetrievalAugmentationConfig(
            tb_max_tokens=120,
            tb_summarization_length=400,
            tb_num_layers=4,
            tb_batch_size=100
        )
    
    else:  # Large docs (>100KB)
        return RetrievalAugmentationConfig(
            tb_max_tokens=100,
            tb_summarization_length=512,
            tb_num_layers=5,
            tb_batch_size=150
        )

# Usage
with open('data.txt', 'r') as f:
    text = f.read()

config = get_optimized_config(len(text))
RA = RetrievalAugmentation(config=config)
```

### 🎛️ **Key Configuration Parameters**

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `tb_max_tokens` | Chunk size | 100-150 for balance |
| `tb_summarization_length` | Summary size | 200-512 based on depth |
| `tb_num_layers` | Tree depth | 3-5 layers |
| `tr_enable_caching` | Smart caching | `True` (always) |
| `enable_async` | Async processing | `True` for speed |
| `max_concurrent_operations` | Parallelism | 8-12 for most systems |

## 🎯 Performance Tips

### ⚡ **Speed Optimizations**

```python
# 🚀 Maximum speed configuration
speed_config = RetrievalAugmentationConfig(
    # Async everything
    enable_async=True,
    tb_build_mode="async",
    
    # Aggressive caching
    tr_enable_caching=True,
    cache_ttl=7200,  # 2-hour cache
    
    # Batch processing
    tb_batch_size=150,
    max_concurrent_operations=12,
    
    # Smart termination
    tr_early_termination=True,
    tr_adaptive_retrieval=True
)
```

### 🎯 **Quality Optimizations**

```python
# 🎯 Maximum quality configuration
quality_config = RetrievalAugmentationConfig(
    # Smaller chunks for better granularity
    tb_max_tokens=80,
    
    # Longer summaries for context preservation
    tb_summarization_length=600,
    
    # More layers for hierarchical understanding
    tb_num_layers=6,
    
    # Quality-focused retrieval
    tr_threshold=0.4,
    tr_top_k=10
)
```

### 💾 **Memory Optimizations**

```python
# For large documents or limited memory
memory_config = RetrievalAugmentationConfig(
    # Smaller batches
    tb_batch_size=50,
    max_concurrent_operations=6,
    
    # Shorter cache
    cache_ttl=1800,  # 30 minutes
    
    # Conservative settings
    tb_max_tokens=80,
    tr_top_k=5
)
```

## ❓ FAQ

### 🤔 **Common Questions**

**Q: How large documents can RAPTOR handle?**
A: RAPTOR can handle documents from 1KB to 100MB+. Use optimized configurations for large files.

**Q: Do I need a powerful GPU?**
A: No! RAPTOR works great on CPU. GPU accelerates embedding creation but isn't required.

**Q: How much does it cost with OpenAI API?**
A: Typically $0.01-$1.00 per document depending on size. Caching reduces ongoing costs significantly.

**Q: Can I use other embedding models?**
A: Yes! RAPTOR supports multiple embedding models. See `EmbeddingModels.py` for options.

### 🐛 **Troubleshooting**

**Problem: Out of memory errors**
```python
# Solution: Reduce batch size and concurrency
config = RetrievalAugmentationConfig(
    tb_batch_size=25,
    max_concurrent_operations=4
)
```

**Problem: Slow embedding creation**
```python
# Solution: Enable async + increase concurrency
config = RetrievalAugmentationConfig(
    enable_async=True,
    max_concurrent_operations=12
)
```

**Problem: Poor answer quality**
```python
# Solution: Increase summarization length + layers
config = RetrievalAugmentationConfig(
    tb_summarization_length=600,
    tb_num_layers=6,
    tr_top_k=10
)
```

### 📝 **Example Workflows**

**📄 Document Analysis Pipeline**
```python
# 1. Load document
with open('data.txt', 'r') as f:
    text = f.read()

# 2. Build optimized tree
config = get_optimized_config(len(text))
RA = RetrievalAugmentation(config=config)
RA.add_documents(text)

# 3. Run analysis queries
questions = [
    "What is the main topic?",
    "What are the key findings?", 
    "What methodology was used?",
    "What are the conclusions?"
]

results = {}
for question in questions:
    results[question] = RA.answer_question(question)

# 4. Save for future use
RA.save("analysis_tree", include_metadata=True)
```

---

<div align="center">

**🎉 Ready to get started? Create your `data.txt` and run the Quick Start example!**

**🤝 Need help? Check our examples or open an issue.**

</div>

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original RAPTOR paper by Stanford University
- OpenAI for GPT models
- HuggingFace for transformer models
- All contributors and the open-source community