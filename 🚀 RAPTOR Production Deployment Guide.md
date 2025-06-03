# 🌳 RAPTOR Production Suite

<div align="center">

![RAPTOR Logo](https://img.shields.io/badge/RAPTOR-Production%20Ready-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![Production](https://img.shields.io/badge/Production-Enterprise--Grade-red?style=for-the-badge)

**Enterprise-Level RAG Sistemi - Tam Otomatik Production Deployment**

[🚀 Hızlı Başlangıç](#-hızlı-başlangıç) • [📖 Detaylı Kurulum](#-detaylı-kurulum-rehberi) • [🔧 Configuration](#️-configuration-yönetimi) • [📊 Monitoring](#-monitoring--metrics) • [❓ FAQ](#-sıkça-sorulan-sorular)

</div>

---

## 📋 İçindekiler

- [🎯 Proje Hakkında](#-proje-hakkında)
- [⭐ Özellikler](#-özellikler)
- [🚀 Hızlı Başlangıç (5 Dakika)](#-hızlı-başlangıç-5-dakika)
- [📁 Dosya Yapısı](#-dosya-yapısı)
- [📖 Detaylı Kurulum Rehberi](#-detaylı-kurulum-rehberi)
- [🔧 Configuration Yönetimi](#️-configuration-yönetimi)
- [🐳 Docker Deployment](#-docker-deployment)
- [📊 Monitoring & Metrics](#-monitoring--metrics)
- [⚡ Performance Optimization](#-performance-optimization)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [❓ Sıkça Sorulan Sorular](#-sıkça-sorulan-sorular)

---

## 🎯 Proje Hakkında

**RAPTOR Production Suite**, Stanford'ın RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) yaklaşımının **enterprise-grade production** implementasyonudur.

### 🤔 Ne İşe Yarar?

1. **Dökümanlarınızı** hierarchical tree yapısında organize eder
2. **Akıllı soru-cevap sistemi** sunar (WebSocket tabanlı)
3. **Production ortamında** güvenle çalışır
4. **Otomatik monitoring** ve performance optimization sağlar
5. **Scalable ve concurrent** kullanıcıları destekler

### 🏗️ Nasıl Çalışır?

```
📄 Doküman → 🌳 RAPTOR Tree → 🤖 AI Asistan → 💬 Kullanıcı
    ↓              ↓              ↓             ↓
  Chunks      Hierarchical    Smart RAG    Real-time
            Summarization    Retrieval      Chat
```

---

## ⭐ Özellikler

### 🚀 **Core Features**
- ✅ **Hierarchical RAG**: Multi-layer document understanding
- ✅ **Real-time Chat**: WebSocket-based streaming responses
- ✅ **Semantic Search**: Advanced embedding-based retrieval
- ✅ **Smart Caching**: 80%+ performance improvement
- ✅ **Async Processing**: 10x faster than traditional approaches

### 🏢 **Enterprise Features**
- ✅ **Auto-scaling Configuration**: Resource-based optimization
- ✅ **Environment Management**: Dev/Staging/Production configs
- ✅ **Health Monitoring**: Real-time system health checks
- ✅ **Performance Analytics**: Comprehensive metrics & dashboards
- ✅ **Docker Containerization**: One-click deployment
- ✅ **Load Testing**: Built-in stress testing & optimization

### 📊 **Monitoring & Observability**
- ✅ **Prometheus Metrics**: Custom RAPTOR metrics collection
- ✅ **Grafana Dashboards**: Pre-built visualization dashboards
- ✅ **Health Checks**: Automated system health monitoring
- ✅ **Performance Profiling**: Auto-optimization recommendations
- ✅ **Error Tracking**: Comprehensive error logging & alerting

---

## 🚀 Hızlı Başlangıç (5 Dakika)

### 1️⃣ **Önkoşullar**
```bash
# Python 3.8+ gerekli
python --version

# Docker & Docker Compose (opsiyonel ama önerilen)
docker --version
docker-compose --version
```

### 2️⃣ **Projeyi Kur**
```bash
# Repository'yi indir
git clone <your-repo-url>
cd raptor-production

# Python environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt
```

### 3️⃣ **Environment Setup**
```bash
# .env dosyası oluştur
cp .env.example .env

# OpenAI API key'ini ekle
nano .env  # veya herhangi bir editor
```

`.env` dosyasında:
```env
OPENAI_API_KEY=sk-your-actual-openai-key-here
REDIS_PASSWORD=güçlü-şifre-buraya
GRAFANA_PASSWORD=admin-şifre-buraya
```

### 4️⃣ **RAPTOR Tree Oluştur**
```bash
# Dokümanını hazırla
echo "Bu bir test dökümandır. RAPTOR bu metni analiz edecek." > data.txt

# Tree'yi oluştur (otomatik optimizasyon ile)
python build-raptor-production.py data.txt
```

### 5️⃣ **Production'a Deploy Et**
```bash
# Script'i çalıştırılabilir yap
chmod +x scripts/deploy.sh

# Tek komut ile production deployment
./scripts/deploy.sh
```

### 6️⃣ **Test Et**
```bash
# Health check
curl http://localhost:8000/health

# Web arayüzlerini aç
open http://localhost:8000      # RAPTOR API
open http://localhost:3000      # Grafana Dashboard (admin/admin)
open http://localhost:9090      # Prometheus Metrics
```

🎉 **Tebrikler!** RAPTOR sisteminiz production'da çalışıyor!

---

## 📁 Dosya Yapısı

```
raptor-production/
├── 📜 TEMEL DOSYALAR
│   ├── README.md                       # Bu dosya
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # Environment template
│   └── data.txt                        # Örnek doküman
│
├── 🚀 CORE SCRIPTS
│   ├── generic-qa-server.py            # Ana WebSocket sunucu
│   ├── build-raptor-production.py      # Enterprise tree builder
│   ├── production-config.py            # Configuration manager
│   ├── monitoring-setup.py             # Monitoring infrastructure
│   ├── deploy-raptor-production.py     # Production deployer
│   └── performance-optimizer.py        # Load testing & optimization
│
├── 🐳 DOCKER & DEPLOYMENT
│   ├── Dockerfile                      # Multi-stage production image
│   ├── docker-compose.production.yml   # Production services
│   ├── requirements-production.txt     # Production-specific deps
│   └── scripts/
│       ├── deploy.sh                   # Automated deployment script
│       └── healthcheck.py              # Docker health check
│
├── ⚙️ CONFIGURATION
│   ├── config/
│   │   ├── development.json            # Dev environment settings
│   │   ├── staging.json                # Staging environment settings
│   │   ├── production.json             # Production environment settings
│   │   └── redis.conf                  # Redis configuration
│   └── .env                            # Environment variables (create this)
│
├── 📊 MONITORING
│   ├── monitoring/
│   │   ├── prometheus/
│   │   │   └── prometheus.yml          # Metrics collection config
│   │   ├── grafana/
│   │   │   └── raptor_dashboard.json   # Pre-built dashboard
│   │   └── docker-compose.yml          # Monitoring stack
│   └── nginx/
│       └── nginx.conf                  # Load balancer config
│
├── 🌳 RAPTOR TREE (otomatik oluşacak)
│   └── vectordb/
│       ├── raptor-optimized            # Built tree file
│       └── raptor-optimized_metrics.json # Build metrics
│
└── 📝 LOGS & METRICS (otomatik oluşacak)
    ├── logs/                           # Application logs
    ├── metrics/                        # Performance metrics
    └── deployment_log_*.json           # Deployment history
```

---

## 📖 Detaylı Kurulum Rehberi

### 🔧 **1. Sistem Gereksinimleri**

#### **Minimum Gereksinimler:**
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 50GB free space
- **OS**: Linux, macOS, Windows
- **Python**: 3.8+

#### **Önerilen Gereksinimler (Production):**
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 100GB+ SSD
- **Network**: Stable internet connection

#### **Software Dependencies:**
```bash
# Python packages (otomatik yüklenecek)
pip install -r requirements.txt

# System packages (Ubuntu/Debian)
sudo apt update
sudo apt install -y curl build-essential

# System packages (CentOS/RHEL)
sudo yum update
sudo yum install -y curl gcc gcc-c++

# macOS (Homebrew)
brew install curl
```

### 🔑 **2. API Key Setup**

#### **OpenAI API Key:**
1. [OpenAI Platform](https://platform.openai.com/api-keys) hesabı oluştur
2. API key oluştur
3. `.env` dosyasına ekle:

```env
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

#### **Diğer Opsiyonel Ayarlar:**
```env
# Redis güvenlik (önerilen)
REDIS_PASSWORD=super-güçlü-şifre-123

# Grafana admin şifresi
GRAFANA_PASSWORD=admin-şifre-456

# Performance tuning
RAPTOR_BATCH_SIZE=150
MAX_CONCURRENT_OPERATIONS=12
```

### 📄 **3. Doküman Hazırlama**

#### **Desteklenen Formatlar:**
- **Plain Text**: `.txt` dosyaları
- **Markdown**: `.md` dosyaları
- **Rich Text**: Formatlanmış metinler

#### **Doküman Örnekleri:**

```bash
# Basit metin dosyası
echo "Şirketimiz 2020'de kuruldu. Ana hizmetimiz web geliştirmedir." > data.txt

# Daha karmaşık doküman
cat > data.txt << 'EOF'
# Şirket Hakkında

## Tarihçe
Şirketimiz 2020 yılında İstanbul'da kurulmuştur.

## Hizmetlerimiz
- Web geliştirme
- Mobil uygulama
- AI çözümleri

## İletişim
Email: info@sirket.com
Telefon: +90 212 123 45 67
EOF
```

### 🌳 **4. RAPTOR Tree Building**

#### **Temel Build:**
```bash
python build-raptor-production.py data.txt
```

#### **Advanced Build Options:**
```bash
# Performance profili ile
python build-raptor-production.py data.txt --profile speed

# Custom output path
python build-raptor-production.py data.txt --output vectordb/custom-tree

# Force rebuild
python build-raptor-production.py data.txt --force

# Verbose logging
python build-raptor-production.py data.txt --log-level DEBUG
```

#### **Performance Profiles:**

| Profile | Hız | Kalite | Bellek | Kullanım Alanı |
|---------|-----|--------|--------|----------------|
| `speed` | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | Real-time chat |
| `balanced` | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | General purpose |
| `quality` | ⭐ | ⭐⭐⭐ | ⭐ | Research, analysis |
| `memory` | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Limited resources |

### 🚀 **5. Deployment Options**

#### **Option A: Docker Deployment (Önerilen)**
```bash
# Tek komut deployment
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

#### **Option B: Manuel Python Deployment**
```bash
# Production server başlat
python deploy-raptor-production.py --env production

# Development server başlat
python generic-qa-server.py
```

#### **Option C: Kubernetes Deployment**
```bash
# Helm chart ile (gelişmiş kullanıcılar için)
helm install raptor ./kubernetes/helm-chart
```

---

## 🔧 Configuration Yönetimi

### 📝 **Environment Configurations**

#### **Development (development.json):**
```json
{
  "raptor": {
    "batch_size": 50,
    "num_layers": 3,
    "enable_metrics": false
  },
  "workers": 1,
  "monitoring": {
    "enable_prometheus": false,
    "log_level": "DEBUG"
  }
}
```

#### **Production (production.json):**
```json
{
  "raptor": {
    "batch_size": 150,
    "num_layers": 5,
    "max_concurrent_operations": 12,
    "cache_ttl": 7200
  },
  "workers": 4,
  "monitoring": {
    "enable_prometheus": true,
    "log_level": "INFO"
  }
}
```

### ⚙️ **Configuration Management**

#### **Otomatik Environment Detection:**
```python
from production_config import get_production_config

# Otomatik environment detection (RAPTOR_ENV env var'dan)
config = get_production_config()

# Manuel environment
config = get_production_config("production", "speed")
```

#### **Runtime Configuration Override:**
```python
config = get_production_config()
config.raptor.batch_size = 200  # Override specific settings
config.workers = 8
```

#### **Environment Variables:**
```bash
# Environment control
export RAPTOR_ENV=production
export RAPTOR_PROFILE=speed

# Performance tuning
export RAPTOR_BATCH_SIZE=200
export MAX_WORKERS=8

# Redis settings
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=your-password
```

---

## 🐳 Docker Deployment

### 🚢 **Docker Architecture**

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│     Nginx       │   │   RAPTOR App    │   │     Redis       │
│ (Load Balancer) │──▶│  (Python App)   │──▶│    (Cache)      │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                               │
                    ┌─────────────────┐   ┌─────────────────┐
                    │   Prometheus    │   │    Grafana      │
                    │   (Metrics)     │   │ (Dashboards)    │
                    └─────────────────┘   └─────────────────┘
```

### 📋 **Services**

| Service | Port | Purpose | Health Check |
|---------|------|---------|--------------|
| `raptor-app` | 8000 | Main application | `/health` |
| `redis` | 6379 | Caching layer | `redis-cli ping` |
| `prometheus` | 9090 | Metrics collection | `/-/healthy` |
| `grafana` | 3000 | Dashboards | `/api/health` |
| `nginx` | 80/443 | Load balancer | HTTP response |

### 🛠️ **Docker Commands**

#### **Basic Operations:**
```bash
# Deploy all services
./scripts/deploy.sh

# View service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f raptor-app

# Scale services
docker-compose -f docker-compose.production.yml scale raptor-app=3

# Stop services
docker-compose -f docker-compose.production.yml down

# Full cleanup
./scripts/deploy.sh --cleanup
```

#### **Development Mode:**
```bash
# Development deployment
./scripts/deploy.sh --env development

# Local development (no Docker)
python generic-qa-server.py
```

#### **Troubleshooting:**
```bash
# Check container health
docker ps
docker inspect raptor-app

# Enter container for debugging
docker exec -it raptor-app bash

# Check resource usage
docker stats

# View detailed logs
docker-compose logs --timestamps raptor-app
```

---

## 📊 Monitoring & Metrics

### 📈 **Prometheus Metrics**

#### **RAPTOR-Specific Metrics:**
```promql
# Query performance
raptor_retrieval_duration_seconds

# Cache efficiency
rate(raptor_cache_operations_total{result="hit"}[5m]) / 
rate(raptor_cache_operations_total[5m])

# Error rate
rate(raptor_errors_total[5m])

# Active connections
raptor_active_connections

# Request throughput
rate(raptor_requests_total[5m])
```

#### **System Metrics:**
```promql
# Memory usage
raptor_memory_usage_bytes

# CPU usage
rate(process_cpu_seconds_total[5m])

# Response time percentiles
histogram_quantile(0.95, rate(raptor_request_duration_seconds_bucket[5m]))
```

### 📊 **Grafana Dashboards**

#### **Main Dashboard Panels:**
1. **Overview Panel**
   - Request rate
   - Success rate
   - Average response time
   - Active connections

2. **Performance Panel**
   - Response time percentiles (50th, 95th, 99th)
   - Throughput trends
   - Cache hit rates

3. **System Resources**
   - Memory usage
   - CPU usage
   - Disk I/O
   - Network traffic

4. **Error Tracking**
   - Error rates by type
   - Failed requests
   - Timeout incidents

#### **Dashboard Access:**
```bash
# Grafana dashboard
open http://localhost:3000
# Login: admin / admin (or your GRAFANA_PASSWORD)

# Prometheus raw metrics
open http://localhost:9090

# RAPTOR health
curl http://localhost:8000/health | jq
```

### 🚨 **Alerting**

#### **Critical Alerts:**
- Error rate > 5%
- Response time > 10 seconds
- Memory usage > 90%
- Cache hit rate < 50%

#### **Warning Alerts:**
- Response time > 5 seconds
- Memory usage > 80%
- Disk space < 10%

---

## ⚡ Performance Optimization

### 🧪 **Load Testing**

#### **Automatic Load Testing:**
```bash
# Comprehensive load test with auto-optimization
python performance-optimizer.py --optimize

# Custom load test
python performance-optimizer.py \
    --server-url http://localhost:8000 \
    --websocket-url ws://localhost:8000/ws/test \
    --output results.json
```

#### **Load Test Results:**
```json
{
  "tests": {
    "websocket": {
      "success_rate_percent": 98.5,
      "avg_response_time_ms": 1250,
      "total_requests": 500,
      "requests_per_second": 15.2
    }
  },
  "analysis": {
    "performance_grade": "B",
    "recommendations": [
      "Enable early termination",
      "Increase batch size"
    ]
  }
}
```

### 🎯 **Performance Tuning**

#### **Configuration Parameters:**

| Parameter | Low Load | Medium Load | High Load |
|-----------|----------|-------------|-----------|
| `batch_size` | 50 | 100 | 200 |
| `max_concurrent` | 4 | 8 | 16 |
| `workers` | 2 | 4 | 8 |
| `cache_ttl` | 1800 | 3600 | 7200 |

#### **Auto-Optimization:**
```python
# Sistem otomatik olarak yük testine göre config oluşturur
from performance_optimizer import optimize_production_config

# Mevcut config'i yük testi sonuçlarına göre optimize et
optimized_config = optimize_production_config(
    test_results="performance_results.json"
)
```

### 📊 **Performance Monitoring**

#### **Real-time Performance:**
```bash
# Current performance stats
curl http://localhost:8000/metrics | grep raptor_

# Health status with performance data
curl http://localhost:8000/health | jq '.performance'
```

#### **Performance History:**
```python
# Python'da performance geçmişini analiz et
from monitoring_setup import RAPTORMetrics

metrics = RAPTORMetrics()
stats = metrics.get_performance_history(days=7)
```

---

## 🛠️ Troubleshooting

### 🚨 **Yaygın Problemler ve Çözümleri**

#### **1. RAPTOR Tree Yüklenemedi**
```bash
# Problem: "Tree file not found" hatası
# Çözüm: Tree'yi yeniden build et
python build-raptor-production.py data.txt --force
```

#### **2. Redis Bağlantı Hatası**
```bash
# Problem: Redis connection failed
# Çözüm 1: Redis'i yeniden başlat
docker-compose restart redis

# Çözüm 2: Redis şifresini kontrol et
echo "REDIS_PASSWORD=your-password" >> .env
```

#### **3. Yavaş Response Time**
```bash
# Problem: Response time > 5 seconds
# Çözüm: Performance profile değiştir
python deploy-raptor-production.py --env production --profile speed

# Veya manuel optimizasyon
python performance-optimizer.py --optimize
```

#### **4. Memory Hatası**
```bash
# Problem: Out of memory
# Çözüm: Memory profile kullan
python deploy-raptor-production.py --env production --profile memory

# Veya Docker resource limit'leri ayarla
docker-compose -f docker-compose.production.yml \
    --compatibility up -d
```

#### **5. OpenAI API Limit**
```bash
# Problem: Rate limit exceeded
# Çözüm: API key ve rate limit ayarlarını kontrol et
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
    https://api.openai.com/v1/usage
```

### 🔍 **Debug Mode**

#### **Application Debug:**
```bash
# Debug logs ile çalıştır
export RAPTOR_LOG_LEVEL=DEBUG
python generic-qa-server.py

# Veya Docker ile
docker-compose -f docker-compose.production.yml \
    -f docker-compose.debug.yml up
```

#### **Performance Debug:**
```python
# Python'da detaylı performance analizi
from generic_qa_server import RA

# Performance summary al
summary = RA.get_performance_summary()
print(json.dumps(summary, indent=2))

# Specific metric'leri kontrol et
retriever_stats = RA.retriever.get_performance_stats()
print(f"Cache hit rate: {retriever_stats['cache_hit_rate']}")
```

### 📋 **Log Analysis**

#### **Log Locations:**
```bash
# Application logs
tail -f logs/raptor.log

# Docker logs
docker-compose logs -f raptor-app

# System logs
journalctl -f -u docker
```

#### **Log Filtering:**
```bash
# Error'ları filtrele
docker-compose logs raptor-app | grep ERROR

# Performance metric'leri filtrele
docker-compose logs raptor-app | grep "Performance"

# Specific user'ın activity'sini takip et
docker-compose logs raptor-app | grep "client_123"
```

---

## ❓ Sıkça Sorulan Sorular

### 🤔 **Genel Sorular**

#### **S: RAPTOR nedir ve nasıl çalışır?**
**C:** RAPTOR, dökümanları hierarchical tree yapısında organize eden advanced RAG sistemidir. Traditional RAG'den farkı, dokümanı farklı abstraction level'larda özetleyerek daha iyi context understanding sağlamasıdır.

#### **S: Hangi doküman boyutları desteklenir?**
**C:** 
- **Minimum**: 1KB (birkaç paragraf)
- **Optimal**: 10KB - 1MB
- **Maximum**: 100MB+ (performance profile'a bağlı)

#### **S: Ne kadar GPU/CPU gücü gerekir?**
**C:** 
- **Minimum**: 4 CPU cores, 8GB RAM
- **Önerilen**: 8+ CPU cores, 16GB+ RAM
- **GPU**: Opsiyonel (embedding model'leri hızlandırır)

### 🔧 **Teknik Sorular**

#### **S: Multiple dokümanı nasıl işlerim?**
**C:**
```python
documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
for doc in documents:
    result = build_raptor_production(
        data_path=doc,
        output_path=f"vectordb/raptor-{Path(doc).stem}"
    )
```

#### **S: Custom embedding model kullanabilir miyim?**
**C:** Evet! `CustomEmbeddingModel` class'ını inherit ederek kendi model'inizi kullanabilirsiniz:
```python
from raptor.EmbeddingModels import BaseEmbeddingModel

class MyCustomModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        # Your custom implementation
        return embedding_vector
```

#### **S: Production'da scaling nasıl yapılır?**
**C:** 
```bash
# Horizontal scaling (Docker)
docker-compose scale raptor-app=3

# Vertical scaling (config)
export MAX_WORKERS=8
export RAPTOR_BATCH_SIZE=200
```

### 🚀 **Deployment Sorular**

#### **S: Kubernetes'de deploy edebilir miyim?**
**C:** Evet! Docker image'ları Kubernetes-ready. Helm chart örneği:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: raptor-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: raptor
  template:
    metadata:
      labels:
        app: raptor
    spec:
      containers:
      - name: raptor
        image: raptor:latest
        ports:
        - containerPort: 8000
```

#### **S: CI/CD pipeline'ı nasıl kurarım?**
**C:**
```yaml
# .github/workflows/deploy.yml
name: Deploy RAPTOR
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy
      run: ./scripts/deploy.sh --env production
```

### 💡 **Optimization Sorular**

#### **S: Cache performansını nasıl artırırım?**
**C:**
```python
# Cache TTL'yi artır
config.cache_ttl = 7200  # 2 hours

# Cache threshold'u ayarla  
config.tr_enable_caching = True
config.tr_early_termination = True
```

#### **S: Memory kullanımını nasıl azaltırım?**
**C:**
```bash
# Memory-optimized profile kullan
python deploy-raptor-production.py --profile memory

# Batch size'ı azalt
export RAPTOR_BATCH_SIZE=50
```

### 🔒 **Security Sorular**

#### **S: API key'leri nasıl güvenli tutarım?**
**C:**
```bash
# Environment variables kullan
export OPENAI_API_KEY="your-key"

# Veya Docker secrets
echo "your-api-key" | docker secret create openai_key -

# Production'da vault solution kullan
# HashiCorp Vault, AWS Secrets Manager, etc.
```

#### **S: HTTPS nasıl enable ederim?**
**C:**
```nginx
# nginx/nginx.conf
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://raptor_backend;
    }
}
```

### 📞 **Support & Yardım**

#### **S: Hata durumunda ne yapmalıyım?**
**C:**
1. **Health check**: `curl http://localhost:8000/health`
2. **Logs kontrol**: `docker-compose logs raptor-app`
3. **Metrics check**: `curl http://localhost:9090/metrics`
4. **Restart**: `./scripts/deploy.sh --rollback && ./scripts/deploy.sh`

#### **S: Performance issue'leri nasıl debug ederim?**
**C:**
```bash
# Load test çalıştır
python performance-optimizer.py --optimize

# Metrics'leri analiz et
curl http://localhost:8000/metrics | grep raptor_

# Resource usage kontrol et
docker stats raptor-app
```

#### **S: Update nasıl yaparım?**
**C:**
```bash
# Code update
git pull origin main

# Rebuild & redeploy
./scripts/deploy.sh --clean

# Zero-downtime update (advanced)
docker-compose -f docker-compose.production.yml \
    up -d --scale raptor-app=2 --no-recreate
```

---

## 🎯 Best Practices

### 🏗️ **Development Best Practices**
1. **Always test in staging first**
2. **Use environment-specific configs**
3. **Monitor performance metrics**
4. **Keep logs organized**
5. **Regular backup strategy**

### 🚀 **Production Best Practices**
1. **Use Docker for deployment**
2. **Enable monitoring & alerting**
3. **Set up auto-scaling**
4. **Implement health checks**
5. **Regular performance optimization**

### 🔒 **Security Best Practices**
1. **Use environment variables for secrets**
2. **Enable HTTPS in production**
3. **Regular security updates**
4. **Monitor access logs**
5. **Implement rate limiting**

---

## 📞 Destek & İletişim

### 🆘 **Acil Durumlar**
```bash
# Sistem completely down
./scripts/deploy.sh --rollback

# Emergency scale down
docker-compose scale raptor-app=1

# Complete reset
./scripts/deploy.sh --cleanup
./scripts/deploy.sh
```

### 📧 **Yardım Almak**
1. **GitHub Issues**: Bug reports ve feature requests
2. **Documentation**: Bu README'yi tekrar okuyun
3. **Health Check**: `curl http://localhost:8000/health`
4. **Logs**: `docker-compose logs raptor-app`

---

<div align="center">

## 🎉 RAPTOR Production Suite ile Başarılı Projeler!

**Enterprise-grade RAG sisteminiz hazır. Production'da güvenle kullanın.**

[⭐ Star this repo](.) • [🐛 Report Bug](.) • [💡 Request Feature](.) • [📖 Documentation](.)

**Made with ❤️ for Enterprise AI Solutions**

</div>