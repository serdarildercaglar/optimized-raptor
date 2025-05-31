import logging
import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
from functools import lru_cache
from dataclasses import dataclass
import pickle
import os
from pathlib import Path

import torch
import numpy as np
from openai import OpenAI, AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import torch.nn.functional as F
import aiohttp
import json

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


@dataclass
class EmbeddingBatch:
    """Batch container for embedding operations"""
    texts: List[str]
    embeddings: Optional[List[List[float]]] = None
    batch_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict] = None


class EmbeddingCache:
    """Intelligent caching system for embeddings"""
    
    def __init__(self, cache_dir: str = "embedding_cache", max_size: int = 10000, ttl: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.memory_cache = {}
        self.access_times = {}
        
    def _get_text_hash(self, text: str, model_name: str) -> str:
        """Generate hash for text + model combination"""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self.ttl
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        cache_key = self._get_text_hash(text, model_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            embedding, timestamp = self.memory_cache[cache_key]
            if self._is_cache_valid(timestamp):
                self.access_times[cache_key] = time.time()
                return embedding
            else:
                del self.memory_cache[cache_key]
                del self.access_times[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if self._is_cache_valid(cached_data['timestamp']):
                        embedding = cached_data['embedding']
                        self._add_to_memory_cache(cache_key, embedding)
                        return embedding
                    else:
                        cache_file.unlink()  # Remove expired cache
            except Exception as e:
                logging.warning(f"Cache read error: {e}")
                
        return None
    
    def set(self, text: str, model_name: str, embedding: List[float]):
        """Store embedding in cache"""
        cache_key = self._get_text_hash(text, model_name)
        timestamp = time.time()
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, embedding)
        
        # Add to disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            cached_data = {
                'embedding': embedding,
                'timestamp': timestamp,
                'text_preview': text[:100],  # For debugging
                'model_name': model_name
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            logging.warning(f"Cache write error: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, embedding: List[float]):
        """Add to memory cache with LRU eviction"""
        if len(self.memory_cache) >= self.max_size:
            # Evict least recently used
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.memory_cache[lru_key]
            del self.access_times[lru_key]
        
        self.memory_cache[cache_key] = (embedding, time.time())
        self.access_times[cache_key] = time.time()
    
    def get_batch(self, texts: List[str], model_name: str) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Get batch embeddings, return embeddings and indices of missing items"""
        embeddings = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.get(text, model_name)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                missing_indices.append(i)
        
        return embeddings, missing_indices
    
    def set_batch(self, texts: List[str], model_name: str, embeddings: List[List[float]]):
        """Store batch embeddings"""
        for text, embedding in zip(texts, embeddings):
            self.set(text, model_name, embedding)


class BaseEmbeddingModel(ABC):
    """Enhanced base embedding model with async and batch support"""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = "embedding_cache"):
        self.cache_enabled = cache_enabled
        self.cache = EmbeddingCache(cache_dir) if cache_enabled else None
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """Synchronous embedding creation (backward compatibility)"""
        pass
    
    @abstractmethod
    async def create_embedding_async(self, text: str) -> List[float]:
        """Asynchronous embedding creation"""
        pass
    
    @abstractmethod
    async def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Batch embedding creation"""
        pass
    
    def _check_cache(self, text: str) -> Optional[List[float]]:
        """Check cache for existing embedding"""
        if self.cache_enabled and self.cache:
            return self.cache.get(text, self.model_name)
        return None
    
    def _store_cache(self, text: str, embedding: List[float]):
        """Store embedding in cache"""
        if self.cache_enabled and self.cache:
            self.cache.set(text, self.model_name, embedding)


class AsyncOpenAIEmbeddingModel(BaseEmbeddingModel):
    """Optimized OpenAI embedding model with async and batch support"""
    
    def __init__(self, model: str = "text-embedding-ada-002", 
                 max_batch_size: int = 2048,
                 max_concurrent_requests: int = 10,
                 cache_enabled: bool = True):
        super().__init__(cache_enabled)
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.sync_client = OpenAI()
        self.async_client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.model_name = f"OpenAI_{model}"
    
    def create_embedding(self, text: str) -> List[float]:
        """Synchronous embedding creation with caching"""
        # Check cache first
        cached_embedding = self._check_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Create embedding
        embedding = self._create_embedding_sync(text)
        
        # Store in cache
        self._store_cache(text, embedding)
        
        return embedding
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _create_embedding_sync(self, text: str) -> List[float]:
        """Internal sync embedding creation with retry"""
        text = text.replace("\n", " ")
        response = self.sync_client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
    
    async def create_embedding_async(self, text: str) -> List[float]:
        """Asynchronous embedding creation with caching"""
        # Check cache first
        cached_embedding = self._check_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Create embedding
        async with self.semaphore:
            embedding = await self._create_embedding_async_internal(text)
        
        # Store in cache
        self._store_cache(text, embedding)
        
        return embedding
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    async def _create_embedding_async_internal(self, text: str) -> List[float]:
        """Internal async embedding creation with retry"""
        text = text.replace("\n", " ")
        response = await self.async_client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
    
    async def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Optimized batch embedding creation with caching"""
        if not texts:
            return []
        
        batch_size = batch_size or min(self.max_batch_size, len(texts))
        
        # Check cache for all texts
        if self.cache_enabled and self.cache:
            cached_embeddings, missing_indices = self.cache.get_batch(texts, self.model_name)
            
            if not missing_indices:  # All embeddings cached
                return [emb for emb in cached_embeddings if emb is not None]
            
            # Get embeddings for missing texts only
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = await self._create_embeddings_batch_internal(missing_texts, batch_size)
            
            # Merge cached and new embeddings
            result_embeddings = []
            new_emb_iter = iter(new_embeddings)
            
            for i, cached_emb in enumerate(cached_embeddings):
                if cached_emb is not None:
                    result_embeddings.append(cached_emb)
                else:
                    new_emb = next(new_emb_iter)
                    result_embeddings.append(new_emb)
                    # Cache the new embedding
                    self.cache.set(texts[i], self.model_name, new_emb)
            
            return result_embeddings
        else:
            # No caching, create all embeddings
            return await self._create_embeddings_batch_internal(texts, batch_size)
    
    async def _create_embeddings_batch_internal(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Internal batch embedding creation"""
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        # Split into batches
        batches = [cleaned_texts[i:i + batch_size] for i in range(0, len(cleaned_texts), batch_size)]
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = self._process_embedding_batch(batch)
            tasks.append(task)
        
        # Wait for all batches
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_embeddings = []
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    async def _process_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of embeddings with retry"""
        async with self.semaphore:
            response = await self.async_client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]


class AsyncSBertEmbeddingModel(BaseEmbeddingModel):
    """Optimized Sentence-BERT model with async and batch support"""
    
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
                 cache_enabled: bool = True,
                 max_batch_size: int = 32):
        super().__init__(cache_enabled)
        self.model_name_str = model_name
        self.max_batch_size = max_batch_size
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.model_name = f"SBert_{model_name.split('/')[-1]}"
    
    def create_embedding(self, text: str) -> List[float]:
        """Synchronous embedding creation with caching"""
        cached_embedding = self._check_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        embedding = self.model.encode(text).tolist()
        self._store_cache(text, embedding)
        return embedding
    
    async def create_embedding_async(self, text: str) -> List[float]:
        """Asynchronous embedding creation"""
        # For local models, we can run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.create_embedding, text)
    
    async def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Batch embedding creation optimized for local model"""
        if not texts:
            return []
        
        batch_size = batch_size or min(self.max_batch_size, len(texts))
        
        # Check cache
        if self.cache_enabled and self.cache:
            cached_embeddings, missing_indices = self.cache.get_batch(texts, self.model_name)
            
            if not missing_indices:
                return [emb for emb in cached_embeddings if emb is not None]
            
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = await self._create_embeddings_batch_internal(missing_texts, batch_size)
            
            # Merge results and cache new embeddings
            result_embeddings = []
            new_emb_iter = iter(new_embeddings)
            
            for i, cached_emb in enumerate(cached_embeddings):
                if cached_emb is not None:
                    result_embeddings.append(cached_emb)
                else:
                    new_emb = next(new_emb_iter)
                    result_embeddings.append(new_emb)
                    self.cache.set(texts[i], self.model_name, new_emb)
            
            return result_embeddings
        else:
            return await self._create_embeddings_batch_internal(texts, batch_size)
    
    async def _create_embeddings_batch_internal(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Internal batch processing for local model"""
        loop = asyncio.get_event_loop()
        
        # Process in smaller batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Run batch encoding in executor to avoid blocking
            batch_embeddings = await loop.run_in_executor(
                None, lambda: self.model.encode(batch_texts).tolist()
            )
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


class AsyncCustomEmbeddingModel(BaseEmbeddingModel):
    """Optimized custom embedding model with async and batch support"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large",
                 cache_enabled: bool = True,
                 max_batch_size: int = 16):
        super().__init__(cache_enabled)
        self.model_name_str = model_name
        self.max_batch_size = max_batch_size
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model_name = f"Custom_{model_name.split('/')[-1]}"
    
    def _prepare_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Prepare text with proper prefixes"""
        if isinstance(text, list):
            return [f"passage: {t}" if not t.startswith("query") else t for t in text]
        elif isinstance(text, str):
            return f"passage: {text}" if not text.startswith("query") else text
        return text
    
    def create_embedding(self, text: str) -> List[float]:
        """Synchronous embedding creation with caching"""
        cached_embedding = self._check_cache(text)
        if cached_embedding is not None:
            return cached_embedding
        
        prepared_text = self._prepare_text(text)
        embedding = self.model.encode(prepared_text).tolist()
        self._store_cache(text, embedding)
        return embedding
    
    async def create_embedding_async(self, text: str) -> List[float]:
        """Asynchronous embedding creation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.create_embedding, text)
    
    async def create_embeddings_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Batch embedding creation"""
        if not texts:
            return []
        
        batch_size = batch_size or min(self.max_batch_size, len(texts))
        
        # Check cache
        if self.cache_enabled and self.cache:
            cached_embeddings, missing_indices = self.cache.get_batch(texts, self.model_name)
            
            if not missing_indices:
                return [emb for emb in cached_embeddings if emb is not None]
            
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = await self._create_embeddings_batch_internal(missing_texts, batch_size)
            
            # Merge and cache
            result_embeddings = []
            new_emb_iter = iter(new_embeddings)
            
            for i, cached_emb in enumerate(cached_embeddings):
                if cached_emb is not None:
                    result_embeddings.append(cached_emb)
                else:
                    new_emb = next(new_emb_iter)
                    result_embeddings.append(new_emb)
                    self.cache.set(texts[i], self.model_name, new_emb)
            
            return result_embeddings
        else:
            return await self._create_embeddings_batch_internal(texts, batch_size)
    
    async def _create_embeddings_batch_internal(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Internal batch processing"""
        loop = asyncio.get_event_loop()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            prepared_texts = self._prepare_text(batch_texts)
            
            batch_embeddings = await loop.run_in_executor(
                None, lambda: self.model.encode(prepared_texts).tolist()
            )
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


# Backward compatibility aliases
OpenAIEmbeddingModel = AsyncOpenAIEmbeddingModel
SBertEmbeddingModel = AsyncSBertEmbeddingModel  
CustomEmbeddingModel = AsyncCustomEmbeddingModel


# Utility functions for batch operations
async def create_embeddings_parallel(
    embedding_models: Dict[str, BaseEmbeddingModel],
    texts: List[str],
    batch_size: int = 100
) -> Dict[str, List[List[float]]]:
    """Create embeddings for multiple models in parallel"""
    tasks = {}
    
    for model_name, model in embedding_models.items():
        task = model.create_embeddings_batch(texts, batch_size)
        tasks[model_name] = task
    
    results = await asyncio.gather(*tasks.values())
    
    return dict(zip(tasks.keys(), results))


class EmbeddingPerformanceMonitor:
    """Monitor embedding performance and provide insights"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'batch_requests': 0,
            'total_time': 0.0,
            'api_calls': 0
        }
    
    def record_request(self, cache_hit: bool, batch_size: int, duration: float, api_calls: int = 1):
        """Record request metrics"""
        self.metrics['total_requests'] += 1
        if cache_hit:
            self.metrics['cache_hits'] += 1
        if batch_size > 1:
            self.metrics['batch_requests'] += 1
        self.metrics['total_time'] += duration
        self.metrics['api_calls'] += api_calls
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        total_requests = self.metrics['total_requests']
        if total_requests == 0:
            return self.metrics
        
        cache_hit_rate = self.metrics['cache_hits'] / total_requests
        avg_request_time = self.metrics['total_time'] / total_requests
        batch_usage_rate = self.metrics['batch_requests'] / total_requests
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'avg_request_time': avg_request_time,
            'batch_usage_rate': batch_usage_rate
        }


# Global performance monitor
performance_monitor = EmbeddingPerformanceMonitor()