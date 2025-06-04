
# 🌳 RAPTOR Production Suite

<div align="center">
![RAPTOR Logo](https://img.shields.io/badge/RAPTOR-Production%20Ready-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?style=for-the-badge&logo=fastapi)
![Redis](https://img.shields.io/badge/Redis-Caching-red?style=for-the-badge&logo=redis)

**Enterprise-Grade Hierarchical RAG System**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🔧 Configuration](#%EF%B8%8F-configuration) • [📊 Monitoring](#-monitoring) • [❓ FAQ](#-faq)

</div>
---


---

## 📋 Table of Contents

* [🎯 About](#-about)
* [⭐ Features](#-features)
* [🚀 Quick Start](#-quick-start)
* [📁 Project Structure](#-project-structure)
* [📖 Documentation](#-documentation)
* [🔧 Configuration](#%EF%B8%8F-configuration)
* [📊 Monitoring](#-monitoring)
* [⚡ Performance](#-performance)
* [❓ FAQ](#-faq)

---

## 🎯 About

**RAPTOR Production Suite** is an enterprise-grade implementation of Stanford's RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) approach, designed for production environments with advanced caching, monitoring, and deployment capabilities.

### 🤔 What Does It Do?

1. **Hierarchical Document Processing** : Creates multi-layer tree structures from documents
2. **Intelligent RAG System** : Provides context-aware question answering
3. **Real-time WebSocket API** : Streams responses with tool calls and progress tracking
4. **Production Monitoring** : Comprehensive metrics, health checks, and analytics
5. **Environment-based Configuration** : Development/Staging/Production settings

### 🏗️ How It Works?

```
📄 Documents → 🌳 RAPTOR Tree → 🔍 Smart Retrieval → 🤖 AI Response → 💬 User
    ↓              ↓                ↓                ↓             ↓
  Chunks      Hierarchical     Context-Aware    Streaming      Real-time
            Summarization       Search         Response        Chat
```

---

## ⭐ Features

### 🚀 **Core RAPTOR Features**

* ✅  **Hierarchical Document Understanding** : Multi-layer tree construction with clustering
* ✅  **Async Processing** : Fully asynchronous operations with concurrent handling
* ✅  **Advanced Caching** : Query similarity-based caching with Redis backend
* ✅  **Adaptive Retrieval** : Smart parameter adjustment based on query characteristics
* ✅  **Early Termination** : Performance optimization for faster responses

### 🏢 **Production Features**

* ✅  **Real-time WebSocket API** : Streaming responses with progress tracking
* ✅  **Environment Management** : Development/Staging/Production configurations
* ✅  **Health Monitoring** : System health checks and status endpoints
* ✅  **Performance Analytics** : Real-time metrics and historical analysis
* ✅  **Load Testing** : Built-in performance testing tools
* ✅  **Redis Integration** : Caching and session management

### 📊 **Monitoring & Analytics**

* ✅  **Live Metrics** : Real-time performance monitoring via HTTP endpoints
* ✅  **Historical Data** : Performance trends and analytics
* ✅  **Resource Tracking** : Memory, CPU, and cache efficiency monitoring
* ✅  **Error Tracking** : Error logging and analysis
* ✅  **Cache Management** : Memory and Redis cache monitoring

### 🔧 **Technical Features**

* ✅  **Multiple Embedding Models** : OpenAI, SentenceTransformers, Custom models
* ✅  **Batch Processing** : Optimized embedding and summarization operations
* ✅  **Memory Management** : Cache cleanup and garbage collection
* ✅  **Configuration Management** : Environment-based settings with validation
* ✅  **Docker Support** : Basic containerization support

---

## 🚀 Quick Start

### 1️⃣ **Prerequisites**

```bash
# Python 3.8+ required
python --version

# Required for memory monitoring (optional)
pip install psutil
```

### 2️⃣ **Installation**

```bash
# Clone repository
git clone <your-repo-url>
cd raptor-production

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ **Environment Setup**

```bash
# Create environment file
cp .env.example .env

# Edit with your settings
nano .env
```

Required `.env` variables:

```env
OPENAI_API_KEY=sk-your-actual-openai-key-here
REDIS_PASSWORD=your-secure-password
RAPTOR_ENV=development
```

### 4️⃣ **Build RAPTOR Tree**

```bash
# Prepare your document
echo "Your document content here..." > data.txt

# Build RAPTOR tree with production optimizations
python build-raptor-production.py data.txt --profile balanced
```

### 5️⃣ **Start Server**

```bash
# Development server
python generic-qa-server.py

# Or production deployment with auto Redis setup
python deploy-raptor-production.py --env production
```

### 6️⃣ **Test**

```bash
# Health check
curl http://localhost:8000/health

# Open web interface
open test.html  # In your browser
```

---

## 📁 Project Structure

```
raptor-production/
├── 📜 CORE FILES
│   ├── README.md                       # This file
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # Environment template
│   └── data.txt                        # Sample document
│
├── 🌳 RAPTOR LIBRARY
│   ├── raptor/
│   │   ├── tree_retriever.py           # Main retrieval system
│   │   ├── cluster_tree_builder.py     # Tree construction
│   │   ├── EmbeddingModels.py          # Embedding models
│   │   ├── tree_structures.py          # Tree data structures
│   │   ├── utils.py                    # Utility functions
│   │   ├── cluster_utils.py            # Clustering algorithms
│   │   ├── tree_builder.py             # Base tree builder
│   │   ├── Retrievers.py               # Base retriever interface
│   │   ├── QAModels.py                 # Question answering models
│   │   ├── SummarizationModels.py      # Summarization models
│   │   ├── FaissRetriever.py           # FAISS-based retrieval
│   │   ├── RetrievalAugmentation.py    # Main RA system
│   │   └── __init__.py                 # Package initialization
│
├── 🚀 PRODUCTION SCRIPTS
│   ├── generic-qa-server.py            # Main WebSocket server
│   ├── build-raptor-production.py      # Production tree builder
│   ├── production-config.py            # Configuration manager
│   ├── deploy-raptor-production.py     # Deployment automation
│   └── performance-optimizer.py        # Load testing & optimization
│
├── ⚙️ CONFIGURATION
│   ├── config/                         # Auto-generated config files
│   └── .env                            # Environment variables (create this)
│
├── 📊 WEB INTERFACE & RESULTS
│   ├── test.html                       # Web chat interface
│   └── performance_results.json        # Performance test results
│
└── 🌳 RAPTOR DATA (auto-generated)
    └── vectordb/
        ├── raptor-production           # Built tree file
        └── raptor-production_metrics.json # Build metrics
```

---

## 📖 Documentation

### 🔧 **Building RAPTOR Trees**

#### Basic Build

```python
# Command line
python build-raptor-production.py your_document.txt

# Or programmatically
from build_raptor_production import build_raptor_production

result = build_raptor_production(
    data_path="your_document.txt",
    output_path="vectordb/my-tree"
)
```

#### Advanced Build with Configuration

```python
# Performance profiles: "speed", "balanced", "quality", "memory"
result = build_raptor_production(
    data_path="large_document.txt",
    output_path="vectordb/optimized-tree",
    performance_profile="speed",
    force_rebuild=True
)

print(f"Build time: {result['metrics']['total_time_seconds']:.1f}s")
print(f"Nodes created: {result['tree_stats']['nodes']}")
```

### 🔍 **Using RAPTOR for Retrieval**

#### Basic Usage

```python
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor import CustomEmbeddingModel, GPT41SummarizationModel

# Configuration
config = RetrievalAugmentationConfig(
    embedding_model=CustomEmbeddingModel(),
    summarization_model=GPT41SummarizationModel(),
    enable_caching=True,
    enable_async=True
)

# Load tree
RA = RetrievalAugmentation(tree="vectordb/raptor-production", config=config)

# Retrieve context
context = RA.retrieve("What is this document about?")
print(context)
```

#### Advanced Retrieval

```python
# Async retrieval with custom parameters
import asyncio

async def advanced_retrieve():
    context, layer_info = await RA.retrieve_async(
        "Complex question requiring detailed analysis",
        top_k=10,
        max_tokens=5000,
        collapse_tree=False,
        return_layer_information=True
    )
    return context, layer_info

# Run async
context, layers = asyncio.run(advanced_retrieve())
```

### 🤖 **Question Answering**

#### Basic QA

```python
# Simple question answering
answer = RA.answer_question("What are the main points?")
print(answer)
```

#### Batch Question Answering

```python
# Process multiple questions
questions = [
    "What is the main topic?",
    "What are the key findings?", 
    "What recommendations are made?"
]

# Async batch processing
answers = await RA.answer_questions_batch(questions, batch_size=5)
for i, answer in enumerate(answers):
    print(f"Q{i+1}: {answer}")
```

### 🌐 **WebSocket API Usage**

#### Web Interface

Open `test.html` in your browser for a full-featured chat interface.

#### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def chat_client():
    uri = "ws://localhost:8000/ws/python_client"
  
    async with websockets.connect(uri) as websocket:
        # Send question
        await websocket.send("What are the main topics?")
      
        # Receive streaming response
        async for message in websocket:
            data = json.loads(message)
          
            if data['type'] == 'content_chunk':
                print(data['content'], end='', flush=True)
            elif data['type'] == 'stream_end':
                print("\n--- Response complete ---")
                break

# Run client
asyncio.run(chat_client())
```

---

## 🔧 Configuration

### 📝 **Environment Configurations**

The system automatically creates configuration files in `config/` directory:

#### Development (config/development.json)

```json
{
  "raptor": {
    "batch_size": 50,
    "num_layers": 3,
    "enable_metrics": false,
    "max_concurrent_operations": 4
  },
  "workers": 1,
  "monitoring": {
    "enable_prometheus": false,
    "log_level": "DEBUG"
  }
}
```

#### Production (config/production.json)

```json
{
  "raptor": {
    "batch_size": 150,
    "num_layers": 5,
    "max_concurrent_operations": 12,
    "cache_ttl": 7200,
    "enable_caching": true,
    "adaptive_retrieval": true
  },
  "workers": 4,
  "monitoring": {
    "log_level": "INFO"
  }
}
```

### ⚙️ **Configuration Management**

```python
from production_config import get_production_config

# Auto-detect environment from RAPTOR_ENV
config = get_production_config()

# Specific environment and profile
config = get_production_config("production", "speed")

# Override specific settings
config.raptor.batch_size = 200
config.workers = 8
```

### 🔧 **Performance Profiles**

| Profile      | Speed  | Quality | Memory | Use Case                          |
| ------------ | ------ | ------- | ------ | --------------------------------- |
| `speed`    | ⭐⭐⭐ | ⭐⭐    | ⭐⭐   | Real-time applications            |
| `balanced` | ⭐⭐   | ⭐⭐⭐  | ⭐⭐   | General purpose                   |
| `quality`  | ⭐     | ⭐⭐⭐  | ⭐     | Research & analysis               |
| `memory`   | ⭐⭐   | ⭐⭐    | ⭐⭐⭐ | Resource-constrained environments |

---

## 📊 Monitoring

### 📈 **Available Metrics**

#### System Metrics

* Memory usage and peak consumption
* CPU utilization
* Request throughput and response times
* Error rates and success rates

#### RAPTOR-Specific Metrics

* Cache hit rates and efficiency
* Retrieval performance
* Tree query statistics
* Embedding processing times

#### Session Management

* Active WebSocket connections
* Chat session analytics
* User activity tracking

### 📊 **Metrics Endpoints**

```bash
# Real-time metrics
curl http://localhost:8000/metrics/live

# Historical performance data (last 24h)
curl http://localhost:8000/metrics/historical?time_range=24h

# Performance analytics
curl http://localhost:8000/analytics/performance?time_range=7d

# Memory usage details
curl http://localhost:8000/memory/usage

# Session management
curl http://localhost:8000/sessions/list?status=active&limit=50
```

### 🔧 **Management Endpoints**

```bash
# Clear all caches
curl -X DELETE http://localhost:8000/cache/clear

# Optimize RAPTOR performance
curl -X POST http://localhost:8000/raptor/optimize

# Delete specific chat history
curl -X DELETE http://localhost:8000/chat_history/session_id
```

---

## ⚡ Performance

### 🧪 **Load Testing**

```bash
# Run comprehensive performance test
python performance-optimizer.py \
    --server-url http://localhost:8000 \
    --websocket-url ws://localhost:8000/ws/test_client \
    --output results.json

# Run test and generate optimization
python performance-optimizer.py --optimize
```

### 📊 **Real Performance Results**

Based on actual testing with the included `performance_results.json`:

| Metric                      | HTTP API   | WebSocket Chat |
| --------------------------- | ---------- | -------------- |
| **Success Rate**      | 100%       | 54.5%          |
| **Avg Response Time** | 1.4s       | 0.8s           |
| **Throughput**        | 12.4 req/s | 0.3 conv/s     |
| **P95 Response Time** | 3.4s       | 3.3s           |
| **Memory Usage**      | ~15GB peak | ~15.6GB peak   |

### 📊 **Performance Optimization**

#### Manual Optimization

```python
# High-performance configuration
config = RetrievalAugmentationConfig(
    tb_batch_size=200,                    # Larger batches
    max_concurrent_operations=16,         # More concurrency
    tr_enable_caching=True,               # Enable caching
    tr_early_termination=True,            # Early termination
    cache_ttl=7200,                       # 2-hour cache
    tr_adaptive_retrieval=True            # Adaptive parameters
)
```

#### Memory Optimization

```python
# Memory-optimized configuration
config = RetrievalAugmentationConfig(
    tb_batch_size=50,                     # Smaller batches
    max_concurrent_operations=4,          # Less concurrency
    cache_ttl=1800,                       # 30-minute cache
    tr_top_k=5                           # Fewer results
)
```

---

## ❓ FAQ

### 🤔 **General Questions**

#### **Q: What is RAPTOR and how does it work?**

**A:** RAPTOR creates hierarchical tree structures from documents by recursively clustering and summarizing content at different abstraction levels. This enables better context understanding compared to traditional flat RAG systems.

#### **Q: What document sizes are supported?**

**A:**

* **Minimum** : 1KB (few paragraphs)
* **Optimal** : 10KB - 1MB
* **Maximum** : 100MB+ (depends on configuration)

#### **Q: What are the system requirements?**

**A:**

* **Minimum** : 4 CPU cores, 8GB RAM
* **Recommended** : 8+ CPU cores, 16GB+ RAM
* **GPU** : Optional (speeds up local embedding models)

### 🔧 **Technical Questions**

#### **Q: How do I process multiple documents?**

**A:**

```python
documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
for doc in documents:
    result = build_raptor_production(
        data_path=doc,
        output_path=f"vectordb/tree-{Path(doc).stem}"
    )
```

#### **Q: Can I use custom embedding models?**

**A:** Yes! Inherit from `BaseEmbeddingModel`:

```python
from raptor.EmbeddingModels import BaseEmbeddingModel

class MyCustomModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        # Your implementation
        return embedding_vector
  
    async def create_embedding_async(self, text):
        # Async implementation
        return await self.create_embedding(text)
```

#### **Q: How do I scale for production?**

**A:**

```bash
# Set environment variables
export MAX_WORKERS=8
export RAPTOR_BATCH_SIZE=200
export RAPTOR_ENV=production

# Use production deployment
python deploy-raptor-production.py --env production
```

### 🚀 **Deployment Questions**

#### **Q: How do I monitor performance in production?**

**A:**

```bash
# Real-time system health
curl http://localhost:8000/health

# Live performance metrics
curl http://localhost:8000/metrics/live

# Memory usage analysis
curl http://localhost:8000/memory/usage

# Performance analytics dashboard
curl http://localhost:8000/analytics/performance
```

#### **Q: How do I handle Redis setup?**

**A:** The deployment script automatically handles Redis:

```bash
# Automatic Redis setup and RAPTOR deployment
python deploy-raptor-production.py --env production
```

#### **Q: How do I optimize memory usage?**

**A:**

```bash
# Use memory-optimized profile
python build-raptor-production.py data.txt --profile memory

# Or set environment variables
export RAPTOR_BATCH_SIZE=50
export MAX_CONCURRENT_OPERATIONS=4
```

### 💡 **Performance Questions**

#### **Q: How can I improve response times?**

**A:**

1. **Enable Early Termination** : Use `--profile speed`
2. **Increase Concurrent Operations** : Set `MAX_CONCURRENT_OPERATIONS=16`
3. **Optimize Cache Settings** : Set longer `cache_ttl`
4. **Use Async Operations** : Enable `enable_async=True`

#### **Q: My WebSocket connections are failing. What should I do?**

**A:**

1. **Check Server Health** : `curl http://localhost:8000/health`
2. **Verify Redis** : Ensure Redis is running
3. **Check Logs** : Look for errors in console output
4. **Test Simple Connection** : Use the included `test.html`

### 🔒 **Security Questions**

#### **Q: How do I secure API keys?**

**A:**

```bash
# Use environment variables
export OPENAI_API_KEY="your-key"

# For production, use secrets management
# Never commit keys to version control
```

#### **Q: How do I clear sensitive chat data?**

**A:**

```bash
# Clear all caches
curl -X DELETE http://localhost:8000/cache/clear

# Delete specific session
curl -X DELETE http://localhost:8000/chat_history/session_id
```

### 📞 **Support & Help**

#### **Q: How do I troubleshoot issues?**

**A:**

1. **Check Health** : `curl http://localhost:8000/health`
2. **View Logs** : Check console output for errors
3. **Test Performance** : `python performance-optimizer.py`
4. **Verify Environment** : Check `.env` file settings

#### **Q: The performance tests show poor results. What should I do?**

**A:**

1. **Check System Resources** : Monitor memory and CPU usage
2. **Optimize Configuration** : Try different performance profiles
3. **Verify RAPTOR Tree** : Ensure tree was built successfully
4. **Check Network** : Verify WebSocket connectivity

---

## 🎯 Best Practices

### 🏗️ **Development Best Practices**

1. **Test with small documents first**
2. **Use development environment for testing**
3. **Monitor resource usage during development**
4. **Keep logs for debugging**
5. **Use appropriate performance profiles**

### 🚀 **Production Best Practices**

1. **Enable comprehensive monitoring**
2. **Set up automated health checks**
3. **Use Redis for optimal performance**
4. **Implement proper error handling**
5. **Regular cache cleanup and optimization**

### 🔒 **Security Best Practices**

1. **Use environment variables for secrets**
2. **Implement session management**
3. **Monitor access patterns**
4. **Regular cache cleanup for privacy**
5. **Use secure Redis passwords**

---

<div align="center">
## 🎉 Ready to Build Amazing RAG Applications!

**RAPTOR Production Suite provides a solid foundation for enterprise-grade hierarchical RAG systems.**

[⭐ Star this repo]() • [🐛 Report Bug]() • [💡 Request Feature]() • [📖 Documentation]()

**Built for Production AI Applications**

</div>
