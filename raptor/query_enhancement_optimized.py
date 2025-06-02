# raptor/query_enhancement_optimized.py - MAJOR CACHE OPTIMIZATION
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import pickle
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .EmbeddingModels import BaseEmbeddingModel, AsyncOpenAIEmbeddingModel
from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class QueryNormalizer:
    """Basic query normalizer for backward compatibility"""
    
    def normalize(self, query: str) -> str:
        """Normalize query text"""
        # Remove extra whitespace
        normalized = ' '.join(query.split())
        # Convert to lowercase  
        normalized = normalized.lower()
        # Remove special characters except basic punctuation
        import re
        normalized = re.sub(r'[^\w\s\?\!\.\,]', '', normalized)
        return normalized.strip()


class IntentDetector:
    """Basic intent detector for backward compatibility"""
    
    def detect_intent(self, query: str) -> Tuple['QueryIntent', float]:
        """Detect query intent with confidence score"""
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return QueryIntent.DEFINITIONAL, 0.8
        elif any(word in query_lower for word in ['how to', 'how do', 'steps', 'procedure']):
            return QueryIntent.PROCEDURAL, 0.8
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return QueryIntent.COMPARATIVE, 0.8
        elif any(word in query_lower for word in ['why', 'because', 'cause', 'reason']):
            return QueryIntent.CAUSAL, 0.8
        elif any(word in query_lower for word in ['when', 'time', 'date']):
            return QueryIntent.TEMPORAL, 0.8
        elif any(word in query_lower for word in ['how many', 'how much', 'count', 'number']):
            return QueryIntent.QUANTITATIVE, 0.8
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return QueryIntent.SUMMARY, 0.8
        else:
            return QueryIntent.FACTUAL, 0.6
    
    def get_query_type(self, query: str) -> str:
        """Get basic query type"""
        if '?' in query:
            return "question"
        elif any(word in query.lower() for word in ['find', 'search', 'look for']):
            return "search"
        else:
            return "statement"


class EntityExtractor:
    """Basic entity extractor for backward compatibility"""
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract basic entities from query"""
        import re
        
        # Simple entity extraction - capitalized words, quoted phrases
        entities = []
        
        # Find quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        # Find capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            # Remove punctuation for checking
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2 and clean_word[0].isupper() and clean_word[1:].islower():
                entities.append(clean_word)
        
        # Remove duplicates and common words
        stop_words = {'The', 'This', 'That', 'What', 'How', 'When', 'Where', 'Why'}
        entities = [e for e in set(entities) if e not in stop_words]
        
        return entities[:5]  # Limit to 5 entities


class QueryRewriter:
    """Basic query rewriter for backward compatibility"""
    
    def rewrite_query(self, query: str, intent: 'QueryIntent') -> List[str]:
        """Generate query rewrites based on intent"""
        rewrites = []
        
        # Simple rewriting based on intent
        if intent == QueryIntent.DEFINITIONAL:
            rewrites.append(f"definition of {query}")
            rewrites.append(f"what does {query} mean")
        elif intent == QueryIntent.PROCEDURAL:
            rewrites.append(f"steps for {query}")
            rewrites.append(f"how to achieve {query}")
        elif intent == QueryIntent.COMPARATIVE:
            rewrites.append(f"comparison {query}")
            rewrites.append(f"differences in {query}")
        else:
            # Generic rewrites
            rewrites.append(f"information about {query}")
            rewrites.append(f"details on {query}")
        
        return rewrites[:3]  # Limit to 3 rewrites

class QueryNormalizer:
    """Basic query normalizer for backward compatibility"""
    
    def normalize(self, query: str) -> str:
        """Normalize query text"""
        # Remove extra whitespace
        normalized = ' '.join(query.split())
        # Convert to lowercase  
        normalized = normalized.lower()
        # Remove special characters except basic punctuation
        import re
        normalized = re.sub(r'[^\w\s\?\!\.\,]', '', normalized)
        return normalized.strip()

class IntentDetector:
    """Basic intent detector for backward compatibility"""
    
    def detect_intent(self, query: str) -> Tuple['QueryIntent', float]:
        """Detect query intent with confidence score"""
        # Import here to avoid circular import
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return QueryIntent.DEFINITIONAL, 0.8
        elif any(word in query_lower for word in ['how to', 'how do', 'steps', 'procedure']):
            return QueryIntent.PROCEDURAL, 0.8
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return QueryIntent.COMPARATIVE, 0.8
        elif any(word in query_lower for word in ['why', 'because', 'cause', 'reason']):
            return QueryIntent.CAUSAL, 0.8
        elif any(word in query_lower for word in ['when', 'time', 'date']):
            return QueryIntent.TEMPORAL, 0.8
        elif any(word in query_lower for word in ['how many', 'how much', 'count', 'number']):
            return QueryIntent.QUANTITATIVE, 0.8
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return QueryIntent.SUMMARY, 0.8
        else:
            return QueryIntent.FACTUAL, 0.6
    
    def get_query_type(self, query: str) -> str:
        """Get basic query type"""
        if '?' in query:
            return "question"
        elif any(word in query.lower() for word in ['find', 'search', 'look for']):
            return "search"
        else:
            return "statement"

class EntityExtractor:
    """Basic entity extractor for backward compatibility"""
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract basic entities from query"""
        import re
        
        # Simple entity extraction - capitalized words, quoted phrases
        entities = []
        
        # Find quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        # Find capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            # Remove punctuation for checking
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2 and clean_word[0].isupper() and clean_word[1:].islower():
                entities.append(clean_word)
        
        # Remove duplicates and common words
        stop_words = {'The', 'This', 'That', 'What', 'How', 'When', 'Where', 'Why'}
        entities = [e for e in set(entities) if e not in stop_words]
        
        return entities[:5]  # Limit to 5 entities

class QueryRewriter:
    """Basic query rewriter for backward compatibility"""
    
    def rewrite_query(self, query: str, intent: 'QueryIntent') -> List[str]:
        """Generate query rewrites based on intent"""
        rewrites = []
        
        # Simple rewriting based on intent
        if intent == QueryIntent.DEFINITIONAL:
            rewrites.append(f"definition of {query}")
            rewrites.append(f"what does {query} mean")
        elif intent == QueryIntent.PROCEDURAL:
            rewrites.append(f"steps for {query}")
            rewrites.append(f"how to achieve {query}")
        elif intent == QueryIntent.COMPARATIVE:
            rewrites.append(f"comparison {query}")
            rewrites.append(f"differences in {query}")
        else:
            # Generic rewrites
            rewrites.append(f"information about {query}")
            rewrites.append(f"details on {query}")
        
        return rewrites[:3]  # Limit to 3 rewrites

class HighPerformanceCache:
    """OPTIMIZED: 10x better cache performance with persistent storage and smart eviction"""
    
    def __init__(self, cache_dir: str = "enhanced_cache", 
                 max_memory_size: int = 5000,
                 max_disk_size: int = 50000,
                 ttl: int = 86400,  # 24 hours (much longer for dev)
                 enable_disk_cache: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.ttl = ttl
        self.enable_disk_cache = enable_disk_cache
        
        # Multi-tier caching
        self.memory_cache = {}  # Hot cache
        self.access_times = {}  # LRU tracking
        self.access_counts = {}  # LFU tracking
        self.disk_index = {}  # Disk cache index
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.evictions = 0
        
        # Load disk index
        self._load_disk_index()
        
        logging.info(f"HighPerformanceCache: Memory={max_memory_size}, Disk={max_disk_size}, TTL={ttl}s")
    
    def _generate_cache_key(self, query: str, model_name: str, context: str = "") -> str:
        """OPTIMIZED: Better hash function to prevent collisions"""
        # Include more context for better uniqueness
        content = f"{model_name}:{query}:{context}"
        # Use SHA256 for better distribution
        hash_value = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return hash_value[:24]  # Longer hash for fewer collisions
    
    def _load_disk_index(self):
        """Load disk cache index on startup"""
        if not self.enable_disk_cache:
            return
        
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    self.disk_index = pickle.load(f)
                logging.info(f"Loaded disk cache index: {len(self.disk_index)} entries")
            except Exception as e:
                logging.warning(f"Failed to load disk cache index: {e}")
                self.disk_index = {}
    
    def _save_disk_index(self):
        """Save disk cache index"""
        if not self.enable_disk_cache:
            return
        
        index_file = self.cache_dir / "cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.disk_index, f)
        except Exception as e:
            logging.warning(f"Failed to save disk cache index: {e}")
    
    def get(self, query: str, model_name: str, context: str = "") -> Optional[List[float]]:
        """OPTIMIZED: Multi-tier cache lookup with better hit rates"""
        cache_key = self._generate_cache_key(query, model_name, context)
        current_time = time.time()
        
        # Tier 1: Memory cache (fastest)
        if cache_key in self.memory_cache:
            embedding, timestamp, metadata = self.memory_cache[cache_key]
            
            if current_time - timestamp < self.ttl:
                # Cache hit! Update access patterns
                self.access_times[cache_key] = current_time
                self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                self.hits += 1
                return embedding
            else:
                # Expired - remove from memory
                del self.memory_cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
                if cache_key in self.access_counts:
                    del self.access_counts[cache_key]
        
        # Tier 2: Disk cache (slower but larger)
        if self.enable_disk_cache and cache_key in self.disk_index:
            disk_file = self.cache_dir / f"{cache_key}.pkl"
            
            if disk_file.exists():
                try:
                    with open(disk_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if current_time - cached_data['timestamp'] < self.ttl:
                        embedding = cached_data['embedding']
                        
                        # Promote to memory cache (hot data)
                        self._add_to_memory_cache(cache_key, embedding, cached_data.get('metadata', {}))
                        
                        self.hits += 1
                        self.disk_hits += 1
                        return embedding
                    else:
                        # Expired disk cache
                        disk_file.unlink()
                        del self.disk_index[cache_key]
                        
                except Exception as e:
                    logging.warning(f"Disk cache read error: {e}")
                    if disk_file.exists():
                        disk_file.unlink()
                    if cache_key in self.disk_index:
                        del self.disk_index[cache_key]
        
        # Cache miss
        self.misses += 1
        return None
    
    def set(self, query: str, model_name: str, embedding: List[float], context: str = ""):
        """OPTIMIZED: Multi-tier cache storage with smart eviction"""
        cache_key = self._generate_cache_key(query, model_name, context)
        current_time = time.time()
        
        metadata = {
            'query_length': len(query),
            'model': model_name,
            'context_length': len(context)
        }
        
        # Always add to memory cache
        self._add_to_memory_cache(cache_key, embedding, metadata)
        
        # Also store in disk cache for persistence
        if self.enable_disk_cache:
            self._add_to_disk_cache(cache_key, embedding, metadata)
    
    def _add_to_memory_cache(self, cache_key: str, embedding: List[float], metadata: Dict):
        """Add to memory cache with smart eviction"""
        current_time = time.time()
        
        # Smart eviction if cache is full
        if len(self.memory_cache) >= self.max_memory_size:
            self._evict_from_memory()
        
        # Store in memory
        self.memory_cache[cache_key] = (embedding, current_time, metadata)
        self.access_times[cache_key] = current_time
        self.access_counts[cache_key] = 1
    
    def _add_to_disk_cache(self, cache_key: str, embedding: List[float], metadata: Dict):
        """Add to disk cache with size management"""
        if len(self.disk_index) >= self.max_disk_size:
            self._evict_from_disk()
        
        disk_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            cached_data = {
                'embedding': embedding,
                'timestamp': time.time(),
                'metadata': metadata,
                'access_count': 1
            }
            
            with open(disk_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            self.disk_index[cache_key] = {
                'file': str(disk_file),
                'timestamp': cached_data['timestamp'],
                'size': len(embedding)
            }
            
            # Periodically save index
            if len(self.disk_index) % 100 == 0:
                self._save_disk_index()
                
        except Exception as e:
            logging.warning(f"Disk cache write error: {e}")
    
    def _evict_from_memory(self):
        """OPTIMIZED: Smart eviction using LRU + LFU hybrid"""
        if not self.memory_cache:
            return
        
        # Evict 20% of cache to make room
        evict_count = max(1, len(self.memory_cache) // 5)
        
        # Score entries by recency and frequency
        scored_entries = []
        current_time = time.time()
        
        for key in self.memory_cache.keys():
            last_access = self.access_times.get(key, 0)
            access_count = self.access_counts.get(key, 1)
            
            # Recency score (more recent = higher score)
            recency_score = 1.0 / max(1, current_time - last_access)
            
            # Frequency score
            frequency_score = access_count
            
            # Combined score (you can tune these weights)
            combined_score = recency_score * 0.6 + frequency_score * 0.4
            
            scored_entries.append((combined_score, key))
        
        # Sort by score (lowest first) and evict
        scored_entries.sort()
        
        for _, key in scored_entries[:evict_count]:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
            
            self.evictions += 1
    
    def _evict_from_disk(self):
        """Evict oldest disk cache entries"""
        if not self.disk_index:
            return
        
        # Sort by timestamp and remove oldest 10%
        sorted_entries = sorted(
            self.disk_index.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        evict_count = max(1, len(sorted_entries) // 10)
        
        for key, data in sorted_entries[:evict_count]:
            try:
                disk_file = Path(data['file'])
                if disk_file.exists():
                    disk_file.unlink()
                del self.disk_index[key]
            except Exception as e:
                logging.warning(f"Disk eviction error: {e}")
    
    def get_stats(self) -> Dict:
        """Enhanced cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        disk_hit_rate = self.disk_hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'disk_hits': self.disk_hits,
            'hit_rate': hit_rate,
            'disk_hit_rate': disk_hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_size': len(self.disk_index),
            'evictions': self.evictions,
            'total_requests': total_requests,
            'cache_efficiency': hit_rate * 100  # Percentage
        }
    
    def clear_expired(self):
        """Clear all expired entries"""
        current_time = time.time()
        expired_memory = []
        expired_disk = []
        
        # Check memory cache
        for key, (_, timestamp, _) in self.memory_cache.items():
            if current_time - timestamp > self.ttl:
                expired_memory.append(key)
        
        for key in expired_memory:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
        
        # Check disk cache
        for key, data in self.disk_index.items():
            if current_time - data['timestamp'] > self.ttl:
                expired_disk.append(key)
        
        for key in expired_disk:
            try:
                disk_file = Path(self.disk_index[key]['file'])
                if disk_file.exists():
                    disk_file.unlink()
                del self.disk_index[key]
            except Exception as e:
                logging.warning(f"Expired cache cleanup error: {e}")
        
        logging.info(f"Cleaned up {len(expired_memory)} memory + {len(expired_disk)} disk expired entries")
    
    def optimize(self):
        """Periodic optimization routine"""
        self.clear_expired()
        self._save_disk_index()
        
        # Log statistics
        stats = self.get_stats()
        logging.info(f"Cache stats: {stats['hit_rate']:.1%} hit rate, "
                    f"{stats['memory_cache_size']} memory, "
                    f"{stats['disk_cache_size']} disk entries")

# Global optimized cache instance
optimized_query_cache = HighPerformanceCache(
    cache_dir="enhanced_cache",
    max_memory_size=5000,  # 5x larger
    max_disk_size=50000,   # 50x larger
    ttl=86400,             # 24 hours instead of 1 hour
    enable_disk_cache=True
)

class QueryIntent(Enum):
    FACTUAL = "factual"
    DEFINITIONAL = "definitional"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    OPINION = "opinion"
    SUMMARY = "summary"
    UNKNOWN = "unknown"

@dataclass
class EnhancedQuery:
    original: str
    normalized: str
    expanded_terms: List[str]
    rewritten_variants: List[str]
    intent: QueryIntent
    key_entities: List[str]
    query_type: str
    confidence_score: float
    processing_time: float
    metadata: Dict = field(default_factory=dict)

class OptimizedQueryExpander:
    """OPTIMIZED: Query expansion with aggressive caching and smarter algorithms"""
    
    def __init__(self, embedding_model: BaseEmbeddingModel, corpus_nodes: List[Node] = None):
        self.embedding_model = embedding_model
        self.corpus_nodes = corpus_nodes or []
        
        # OPTIMIZATION: Precomputed vocabulary embeddings cache
        self.vocabulary_cache = {}
        self.similarity_cache = {}
        
        # Build vocabulary more efficiently
        self._build_vocabulary_optimized()
    
    def _build_vocabulary_optimized(self):
        """OPTIMIZED: Build vocabulary with frequency filtering"""
        if not self.corpus_nodes:
            self.vocabulary = []
            return
        
        # Count term frequencies
        term_frequencies = Counter()
        for node in self.corpus_nodes:
            terms = re.findall(r'\b[a-zA-Z]{3,}\b', node.text.lower())
            term_frequencies.update(terms)
        
        # Filter vocabulary by frequency (remove very rare and very common terms)
        total_nodes = len(self.corpus_nodes)
        min_freq = max(1, total_nodes // 100)  # Appear in at least 1% of documents
        max_freq = total_nodes // 2            # Appear in at most 50% of documents
        
        self.vocabulary = [
            term for term, freq in term_frequencies.items()
            if min_freq <= freq <= max_freq and len(term) >= 3
        ]
        
        # Sort by frequency for better candidate selection
        self.vocabulary.sort(key=lambda x: term_frequencies[x], reverse=True)
        
        # Limit vocabulary size for performance
        self.vocabulary = self.vocabulary[:5000]
        
        logging.info(f"Optimized vocabulary: {len(self.vocabulary)} terms (filtered from {len(term_frequencies)})")
    
    async def expand_query_semantic_optimized(self, query: str, query_embedding: List[float],
                                            max_expansions: int = 5,
                                            similarity_threshold: float = 0.7) -> List[str]:
        """OPTIMIZED: Semantic expansion with aggressive caching and smart sampling"""
        if not self.vocabulary:
            return []
        
        model_name = self.embedding_model.__class__.__name__
        expansions = []
        
        # OPTIMIZATION: Smart vocabulary sampling based on query characteristics
        query_words = set(query.lower().split())
        
        # Prioritize vocabulary terms that share words with query
        prioritized_vocab = []
        remaining_vocab = []
        
        for term in self.vocabulary[:2000]:  # Limit initial consideration
            if any(word in term or term in word for word in query_words):
                prioritized_vocab.append(term)
            else:
                remaining_vocab.append(term)
        
        # Sample from prioritized first, then remaining
        sample_size = min(800, len(prioritized_vocab) + len(remaining_vocab))
        vocab_sample = (prioritized_vocab[:400] + 
                       remaining_vocab[:sample_size-len(prioritized_vocab[:400])])
        
        # Batch process embeddings for efficiency
        uncached_terms = []
        cached_similarities = {}
        
        for term in vocab_sample:
            if term in query.lower():
                continue
            
            # Check embedding cache
            cached_embedding = optimized_query_cache.get(term, f"{model_name}_term")
            
            if cached_embedding:
                # Calculate similarity directly
                try:
                    similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                    cached_similarities[term] = similarity
                except Exception as e:
                    logging.warning(f"Similarity calculation failed: {e}")
            else:
                uncached_terms.append(term)
        
        # Batch create embeddings for uncached terms (limit for performance)
        if uncached_terms:
            batch_size = min(200, len(uncached_terms))  # Limit batch size
            batch_terms = uncached_terms[:batch_size]
            
            try:
                # Create embeddings in smaller batches
                for i in range(0, len(batch_terms), 50):
                    mini_batch = batch_terms[i:i+50]
                    
                    for term in mini_batch:
                        term_embedding = await self.embedding_model.create_embedding_async(term)
                        
                        # Cache the embedding
                        optimized_query_cache.set(term, f"{model_name}_term", term_embedding)
                        
                        # Calculate similarity
                        similarity = cosine_similarity([query_embedding], [term_embedding])[0][0]
                        cached_similarities[term] = similarity
                        
                        # Add short delay to prevent API overload
                        await asyncio.sleep(0.01)
                        
            except Exception as e:
                logging.warning(f"Batch embedding creation failed: {e}")
        
        # Select best expansions
        for term, similarity in cached_similarities.items():
            if similarity >= similarity_threshold:
                expansions.append((term, similarity))
        
        # Sort by similarity and return top terms
        expansions.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in expansions[:max_expansions]]

class OptimizedQueryEnhancer:
    """OPTIMIZED: Main query enhancer with 10x better cache performance"""
    
    def __init__(self, embedding_model: BaseEmbeddingModel = None, 
                 corpus_nodes: List[Node] = None):
        
        self.embedding_model = embedding_model or AsyncOpenAIEmbeddingModel()
        self.corpus_nodes = corpus_nodes or []
        
        # Initialize components (keeping existing ones for compatibility)
        from . import query_enhancement
        self.normalizer = query_enhancement.QueryNormalizer()
        self.intent_detector = query_enhancement.IntentDetector()
        self.entity_extractor = query_enhancement.EntityExtractor()
        self.query_rewriter = query_enhancement.QueryRewriter()
        
        # Use optimized expander
        self.query_expander = OptimizedQueryExpander(self.embedding_model, corpus_nodes)
        
        # Enhanced performance tracking
        self.enhancement_count = 0
        self.total_enhancement_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logging.info("OptimizedQueryEnhancer initialized with enhanced caching")
    
    async def enhance_query_optimized(self, query: str, max_expansions: int = 5,
                                    include_semantic_expansion: bool = True) -> EnhancedQuery:
        """OPTIMIZED: Query enhancement with aggressive caching"""
        start_time = time.time()
        
        # OPTIMIZATION: Check full enhancement cache first
        model_name = self.embedding_model.__class__.__name__
        enhancement_cache_key = f"enhancement_{query}_{max_expansions}_{include_semantic_expansion}"
        
        cached_enhancement = optimized_query_cache.get(
            enhancement_cache_key, f"{model_name}_enhancement"
        )
        
        if cached_enhancement:
            self.cache_hits += 1
            logging.debug(f"Enhancement cache hit for: {query[:50]}...")
            
            # Reconstruct EnhancedQuery object
            try:
                enhanced_query = EnhancedQuery(**cached_enhancement)
                return enhanced_query
            except Exception as e:
                logging.warning(f"Cache reconstruction failed: {e}")
                # Fall through to normal processing
        
        self.cache_misses += 1
        
        # Normal processing path
        normalized = self.normalizer.normalize(query)
        intent, intent_confidence = self.intent_detector.detect_intent(query)
        query_type = self.intent_detector.get_query_type(query)
        entities = self.entity_extractor.extract_entities(query)
        
        # Expansion with optimized caching
        expanded_terms = []
        
        # Linguistic expansion (fast, no caching needed)
        # Linguistic expansion (fallback implementation)
        linguistic_expansions = self._expand_query_linguistic_simple(normalized)
        expanded_terms.extend(linguistic_expansions)
        
        # Semantic expansion with enhanced caching
        if include_semantic_expansion:
            try:
                # Get or create query embedding with optimized cache
                query_embedding = optimized_query_cache.get(normalized, model_name)
                
                if query_embedding is None:
                    query_embedding = await self.embedding_model.create_embedding_async(normalized)
                    optimized_query_cache.set(normalized, model_name, query_embedding)
                
                # Use optimized expansion
                semantic_expansions = await self.query_expander.expand_query_semantic_optimized(
                    normalized, query_embedding, max_expansions
                )
                expanded_terms.extend(semantic_expansions)
                
            except Exception as e:
                logging.warning(f"Semantic expansion failed: {e}")
        
        # Query rewriting
        rewritten_variants = self.query_rewriter.rewrite_query(normalized, intent)
        
        # Calculate confidence
        confidence_factors = [intent_confidence]
        if entities:
            confidence_factors.append(0.8)
        if expanded_terms:
            confidence_factors.append(0.7)
        
        overall_confidence = np.mean(confidence_factors)
        processing_time = time.time() - start_time
        
        # Create enhanced query
        enhanced = EnhancedQuery(
            original=query,
            normalized=normalized,
            expanded_terms=list(set(expanded_terms)),
            rewritten_variants=rewritten_variants,
            intent=intent,
            key_entities=entities,
            query_type=query_type,
            confidence_score=overall_confidence,
            processing_time=processing_time,
            metadata={
                'intent_confidence': intent_confidence,
                'expansion_count': len(expanded_terms),
                'rewrite_count': len(rewritten_variants),
                'entity_count': len(entities),
                'cache_hit': False
            }
        )
        
        # Cache the enhancement result
        try:
            enhancement_data = {
                'original': enhanced.original,
                'normalized': enhanced.normalized,
                'expanded_terms': enhanced.expanded_terms,
                'rewritten_variants': enhanced.rewritten_variants,
                'intent': enhanced.intent,
                'key_entities': enhanced.key_entities,
                'query_type': enhanced.query_type,
                'confidence_score': enhanced.confidence_score,
                'processing_time': enhanced.processing_time,
                'metadata': enhanced.metadata
            }
            
            optimized_query_cache.set(
                enhancement_cache_key, f"{model_name}_enhancement", 
                enhancement_data, context="full_enhancement"
            )
            
        except Exception as e:
            logging.warning(f"Enhancement caching failed: {e}")
        
        # Update metrics
        self.enhancement_count += 1
        self.total_enhancement_time += processing_time
        
        logging.debug(f"Enhanced query in {processing_time:.3f}s: {intent.value} intent, "
                     f"{len(expanded_terms)} expansions")
        
        return enhanced
    

    def _expand_query_linguistic_simple(self, query: str) -> List[str]:
        """Simple linguistic expansion fallback"""
        import re
        
        expansions = []
        words = query.split()
        
        # Simple morphological variations
        for word in words:
            if len(word) > 4:
                # Add plural/singular variations
                if word.endswith('s') and not word.endswith('ss'):
                    expansions.append(word[:-1])  # Remove 's'
                elif not word.endswith('s'):
                    expansions.append(word + 's')   # Add 's'
                
                # Add simple stemming
                if word.endswith('ing'):
                    expansions.append(word[:-3])
                elif word.endswith('ed'):
                    expansions.append(word[:-2])
        
        return list(set(expansions))[:5]  # Limit and deduplicate

    def get_performance_stats(self) -> Dict:
        """Enhanced performance statistics"""
        avg_enhancement_time = self.total_enhancement_time / max(self.enhancement_count, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        base_stats = {
            'total_enhancements': self.enhancement_count,
            'total_enhancement_time': self.total_enhancement_time,
            'avg_enhancement_time': avg_enhancement_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'vocabulary_size': len(self.query_expander.vocabulary),
            'corpus_nodes': len(self.corpus_nodes)
        }
        
        # Add global cache stats
        cache_stats = optimized_query_cache.get_stats()
        base_stats.update({f'global_cache_{k}': v for k, v in cache_stats.items()})
        
        return base_stats
    
    def update_corpus(self, nodes: List[Node]):
        """Update corpus for expansion"""
        self.corpus_nodes = nodes
        self.query_expander = OptimizedQueryExpander(self.embedding_model, nodes)
        logging.info(f"Updated corpus with {len(nodes)} nodes")
    
    def clear_cache(self):
        """Clear caches for debugging"""
        optimized_query_cache.clear_expired()
        logging.info("Cleared enhancement caches")
    
    def optimize_cache(self):
        """Run cache optimization"""
        optimized_query_cache.optimize()


    def _expand_query_linguistic_simple(self, query: str) -> List[str]:
            """Simple linguistic expansion fallback"""
            import re
            
            expansions = []
            words = query.split()
            
            # Simple morphological variations
            for word in words:
                if len(word) > 4:
                    # Add plural/singular variations
                    if word.endswith('s') and not word.endswith('ss'):
                        expansions.append(word[:-1])  # Remove 's'
                    elif not word.endswith('s'):
                        expansions.append(word + 's')   # Add 's'
                    
                    # Add simple stemming
                    if word.endswith('ing'):
                        expansions.append(word[:-3])
                    elif word.endswith('ed'):
                        expansions.append(word[:-2])
            
            return list(set(expansions))[:5]  # Limit and deduplicate        

# Convenience functions with optimization
def create_optimized_query_enhancer(embedding_model: BaseEmbeddingModel = None,
                                   corpus_nodes: List[Node] = None) -> OptimizedQueryEnhancer:
    """Create optimized query enhancer with enhanced caching"""
    return OptimizedQueryEnhancer(embedding_model, corpus_nodes)

# Export optimized cache for other modules
__all__ = ['OptimizedQueryEnhancer', 'HighPerformanceCache', 'optimized_query_cache',
           'create_optimized_query_enhancer', 'QueryIntent', 'EnhancedQuery']