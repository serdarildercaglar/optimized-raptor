# raptor/query_enhancement.py
import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict, Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .EmbeddingModels import BaseEmbeddingModel, AsyncOpenAIEmbeddingModel
from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


import time
from typing import Optional, List

class QueryEmbeddingCache:
    """High-performance query embedding cache - CRITICAL PERFORMANCE FIX"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding"""
        cache_key = f"{model_name}:{hash(query)}"
        
        if cache_key in self.cache:
            embedding, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                self.access_times[cache_key] = time.time()
                self.hits += 1
                return embedding
            else:
                del self.cache[cache_key]
                del self.access_times[cache_key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, model_name: str, embedding: List[float]):
        """Cache embedding with LRU eviction"""
        cache_key = f"{model_name}:{hash(query)}"
        
        # LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[cache_key] = (embedding, time.time())
        self.access_times[cache_key] = time.time()
    
    def get_stats(self):
        """Get cache performance stats"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

# Global query embedding cache instance
query_embedding_cache = QueryEmbeddingCache()


class QueryIntent(Enum):
    """Different types of query intents"""
    FACTUAL = "factual"          # What is X? When did Y happen?
    DEFINITIONAL = "definitional"  # Define X, What is the meaning of Y?
    COMPARATIVE = "comparative"    # Compare X and Y, Difference between A and B
    PROCEDURAL = "procedural"     # How to do X? Steps for Y
    CAUSAL = "causal"            # Why does X happen? What causes Y?
    TEMPORAL = "temporal"        # When did X happen? Timeline of Y
    QUANTITATIVE = "quantitative" # How many? How much? Statistics about X
    QUALITATIVE = "qualitative"  # Describe X, What are characteristics of Y
    OPINION = "opinion"          # What do you think about X?
    SUMMARY = "summary"          # Summarize X, Give overview of Y
    UNKNOWN = "unknown"          # Cannot determine intent


@dataclass
class EnhancedQuery:
    """Container for enhanced query information"""
    original: str
    normalized: str
    expanded_terms: List[str]
    rewritten_variants: List[str]
    intent: QueryIntent
    key_entities: List[str]
    query_type: str  # question, statement, keyword
    confidence_score: float
    processing_time: float
    metadata: Dict = field(default_factory=dict)







class QueryNormalizer:
    """Normalize and clean queries"""
    
    def __init__(self):
        # Common query cleaning patterns
        self.patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'[^\w\s\?\!\.]', ' '),  # Remove special chars except basic punctuation
            (r'\b(um|uh|well|like)\b', ''),  # Remove filler words
            (r'\b(please|kindly)\b', ''),  # Remove politeness words
        ]
    
    def normalize(self, query: str) -> str:
        """Normalize query text"""
        normalized = query.strip().lower()
        
        for pattern, replacement in self.patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Clean up extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized


class IntentDetector:
    """Detect query intent using pattern matching and ML"""
    
    def __init__(self):
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(what is|what are|what was|what were)\b',
                r'\b(who is|who are|who was|who were)\b',
                r'\b(where is|where are|where was|where were)\b',
                r'\b(when is|when are|when was|when were|when did)\b',
            ],
            QueryIntent.DEFINITIONAL: [
                r'\b(define|definition of|meaning of|what does.*mean)\b',
                r'\b(explain|explanation of)\b',
                r'\b(what is the meaning)\b',
            ],
            QueryIntent.COMPARATIVE: [
                r'\b(compare|comparison|versus|vs\.?|difference between)\b',
                r'\b(better than|worse than|similar to)\b',
                r'\b(advantages and disadvantages|pros and cons)\b',
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(how to|how do|how can|steps to)\b',
                r'\b(procedure|process|method|way to)\b',
                r'\b(instructions|guide|tutorial)\b',
            ],
            QueryIntent.CAUSAL: [
                r'\b(why|because|reason|cause|causes)\b',
                r'\b(what causes|what leads to|what results in)\b',
                r'\b(due to|as a result of)\b',
            ],
            QueryIntent.TEMPORAL: [
                r'\b(when|timeline|chronology|history of)\b',
                r'\b(before|after|during|since|until)\b',
                r'\b(first|last|recent|latest)\b',
            ],
            QueryIntent.QUANTITATIVE: [
                r'\b(how many|how much|number of|amount of|quantity)\b',
                r'\b(statistics|data|percentage|ratio|rate)\b',
                r'\b(count|total|sum|average|mean)\b',
            ],
            QueryIntent.QUALITATIVE: [
                r'\b(describe|description|characteristics|features)\b',
                r'\b(what.*like|appearance|quality|nature)\b',
                r'\b(properties|attributes|traits)\b',
            ],
            QueryIntent.OPINION: [
                r'\b(what do you think|opinion|believe|feel)\b',
                r'\b(should|ought to|recommend|suggest)\b',
                r'\b(best|worst|favorite|prefer)\b',
            ],
            QueryIntent.SUMMARY: [
                r'\b(summary|summarize|overview|brief|outline)\b',
                r'\b(main points|key points|highlights)\b',
                r'\b(in short|in brief|tell me about)\b',
            ],
        }
    
    def detect_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Detect query intent with confidence score
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        query_lower = query.lower()
        intent_scores = defaultdict(int)
        
        # Pattern-based detection
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_scores[intent] += 1
        
        if not intent_scores:
            return QueryIntent.UNKNOWN, 0.0
        
        # Find best matching intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on pattern matches
        total_matches = sum(intent_scores.values())
        confidence = best_intent[1] / total_matches
        
        return best_intent[0], confidence
    
    def get_query_type(self, query: str) -> str:
        """Determine if query is question, statement, or keywords"""
        query_stripped = query.strip()
        
        if query_stripped.endswith('?'):
            return "question"
        elif len(query_stripped.split()) <= 3:
            return "keyword"
        else:
            return "statement"


class EntityExtractor:
    """Extract key entities from queries"""
    
    def __init__(self):
        # Simple entity patterns (can be enhanced with NER models)
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\d{4}\b',  # Years
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Money
        ]
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates


class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self, embedding_model: BaseEmbeddingModel, corpus_nodes: List[Node] = None):
        self.embedding_model = embedding_model
        self.corpus_nodes = corpus_nodes or []
        self.term_embeddings = {}
        self.similarity_cache = {}
        
        # Build term vocabulary from corpus
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build vocabulary from corpus for expansion"""
        if not self.corpus_nodes:
            return
        
        vocab = set()
        for node in self.corpus_nodes:
            # Extract meaningful terms (simple approach)
            terms = re.findall(r'\b[a-zA-Z]{3,}\b', node.text.lower())
            vocab.update(terms)
        
        self.vocabulary = list(vocab)
        logging.info(f"Built vocabulary with {len(self.vocabulary)} terms")
    
    async def expand_query_semantic(self, query: str, max_expansions: int = 5,
                                  similarity_threshold: float = 0.7) -> List[str]:
        """
        Expand query using semantic similarity
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansion terms
            similarity_threshold: Minimum similarity for expansion
            
        Returns:
            List of expansion terms
        """
        if not self.vocabulary:
            return []
        
        # Get query embedding
        query_embedding = await self.embedding_model.create_embedding_async(query)
        
        # Find similar terms
        expansions = []
        
        # Sample vocabulary for efficiency (you can optimize this)
        sample_size = min(1000, len(self.vocabulary))
        vocab_sample = np.random.choice(self.vocabulary, sample_size, replace=False)
        
        for term in vocab_sample:
            if term in query.lower():
                continue  # Skip terms already in query
            
            # Get term embedding (with caching)
            if term not in self.term_embeddings:
                self.term_embeddings[term] = await self.embedding_model.create_embedding_async(term)
            
            term_embedding = self.term_embeddings[term]
            
            # Calculate similarity
            similarity = cosine_similarity(
                [query_embedding], [term_embedding]
            )[0][0]
            
            if similarity >= similarity_threshold:
                expansions.append((term, similarity))
        
        # Sort by similarity and return top terms
        expansions.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in expansions[:max_expansions]]


    async def _expand_with_cached_embedding(self, query: str, query_embedding: List[float], 
                                        max_expansions: int = 5,
                                        similarity_threshold: float = 0.7) -> List[str]:
        """
        PERFORMANCE FIX: Expand query using pre-computed/cached embedding
        
        Args:
            query: Original query
            query_embedding: Pre-computed query embedding (from cache or fresh)
            max_expansions: Maximum number of expansion terms
            similarity_threshold: Minimum similarity for expansion
            
        Returns:
            List of expansion terms
        """
        if not self.vocabulary:
            return []
        
        # Find similar terms using cached embedding
        expansions = []
        
        # Sample vocabulary for efficiency (you can optimize this)
        sample_size = min(1000, len(self.vocabulary))
        vocab_sample = np.random.choice(self.vocabulary, sample_size, replace=False)
        
        for term in vocab_sample:
            if term in query.lower():
                continue  # Skip terms already in query
            
            # Get term embedding (with caching)
            model_name = self.embedding_model.__class__.__name__
            
            # Check cache for term embedding
            term_embedding = query_embedding_cache.get(term, f"{model_name}_term")
            
            if term_embedding is None:
                # Create and cache term embedding
                term_embedding = await self.embedding_model.create_embedding_async(term)
                query_embedding_cache.set(term, f"{model_name}_term", term_embedding)
            
            # Calculate similarity using cached embeddings
            similarity = cosine_similarity(
                [query_embedding], [term_embedding]
            )[0][0]
            
            if similarity >= similarity_threshold:
                expansions.append((term, similarity))
        
        # Sort by similarity and return top terms
        expansions.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in expansions[:max_expansions]]


    def expand_query_linguistic(self, query: str) -> List[str]:
        """
        Expand query using linguistic rules
        
        Args:
            query: Original query
            
        Returns:
            List of expansion terms
        """
        expansions = []
        
        # Simple synonym mapping (can be enhanced with WordNet or other resources)
        synonym_map = {
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'good': ['excellent', 'great', 'wonderful', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'slow': ['sluggish', 'gradual', 'leisurely'],
            'important': ['significant', 'crucial', 'vital', 'critical'],
            'easy': ['simple', 'straightforward', 'effortless'],
            'hard': ['difficult', 'challenging', 'tough', 'complex'],
        }
        
        words = query.lower().split()
        for word in words:
            if word in synonym_map:
                expansions.extend(synonym_map[word])
        
        return expansions


class QueryRewriter:
    """Rewrite queries to improve retrieval"""
    
    def __init__(self):
        # Template-based rewriting patterns
        self.rewrite_patterns = [
            # Convert questions to statements
            (r'^what is (.+)\?$', r'\1 definition meaning'),
            (r'^who is (.+)\?$', r'\1 person biography'),
            (r'^when did (.+)\?$', r'\1 date time history'),
            (r'^where is (.+)\?$', r'\1 location place'),
            (r'^how to (.+)\?$', r'\1 steps procedure method'),
            (r'^why (.+)\?$', r'\1 reason cause explanation'),
            
            # Add context terms
            (r'\b(define|definition)\b', r'definition meaning explanation'),
            (r'\b(compare|comparison)\b', r'compare difference similarity'),
            (r'\b(history)\b', r'history timeline chronology'),
        ]
    
    def rewrite_query(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Generate rewritten versions of query
        
        Args:
            query: Original query
            intent: Detected query intent
            
        Returns:
            List of rewritten query variants
        """
        variants = []
        
        # Pattern-based rewriting
        for pattern, replacement in self.rewrite_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                rewritten = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                variants.append(rewritten)
        
        # Intent-specific rewriting
        if intent == QueryIntent.FACTUAL:
            variants.append(f"{query} facts information")
        elif intent == QueryIntent.DEFINITIONAL:
            variants.append(f"{query} definition meaning explanation")
        elif intent == QueryIntent.PROCEDURAL:
            variants.append(f"{query} steps process method procedure")
        elif intent == QueryIntent.COMPARATIVE:
            variants.append(f"{query} comparison difference analysis")
        
        # Remove duplicates and original query
        variants = list(set(variants))
        if query in variants:
            variants.remove(query)
        
        return variants


class QueryEnhancer:
    """Main query enhancement orchestrator"""
    
    def __init__(self, embedding_model: BaseEmbeddingModel = None, 
                 corpus_nodes: List[Node] = None):
        
        self.embedding_model = embedding_model or AsyncOpenAIEmbeddingModel()
        self.corpus_nodes = corpus_nodes or []
        
        # Initialize components
        self.normalizer = QueryNormalizer()
        self.intent_detector = IntentDetector()
        self.entity_extractor = EntityExtractor()
        self.query_expander = QueryExpander(self.embedding_model, corpus_nodes)
        self.query_rewriter = QueryRewriter()
        
        # Performance tracking
        self.enhancement_count = 0
        self.total_enhancement_time = 0.0
        
        logging.info("Initialized QueryEnhancer")
    
    async def enhance_query(self, query: str, max_expansions: int = 5,
                          include_semantic_expansion: bool = True) -> EnhancedQuery:
        """
        Comprehensive query enhancement
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansion terms
            include_semantic_expansion: Whether to use semantic expansion
            
        Returns:
            EnhancedQuery object with all enhancements
        """
        start_time = time.time()
        
        # Step 1: Normalize query
        normalized = self.normalizer.normalize(query)
        
        # Step 2: Detect intent
        intent, intent_confidence = self.intent_detector.detect_intent(query)
        query_type = self.intent_detector.get_query_type(query)
        
        # Step 3: Extract entities
        entities = self.entity_extractor.extract_entities(query)
        
        # Step 4: Expand query
        # Step 4: Expand query - WITH PERFORMANCE CACHE
        expanded_terms = []
        
        # Linguistic expansion (fast)
        linguistic_expansions = self.query_expander.expand_query_linguistic(normalized)
        expanded_terms.extend(linguistic_expansions)
        
        # Semantic expansion with caching (PERFORMANCE FIX)
        if include_semantic_expansion:
            try:
                # Check cache first
                model_name = self.embedding_model.__class__.__name__
                cached_embedding = query_embedding_cache.get(normalized, model_name)
                
                if cached_embedding:
                    # Use cached embedding
                    semantic_expansions = await self.query_expander._expand_with_cached_embedding(
                        normalized, cached_embedding, max_expansions
                    )
                else:
                    # Create new embedding and cache it
                    query_embedding = await self.embedding_model.create_embedding_async(normalized)
                    query_embedding_cache.set(normalized, model_name, query_embedding)
                    
                    semantic_expansions = await self.query_expander._expand_with_cached_embedding(
                        normalized, query_embedding, max_expansions
                    )
                
                expanded_terms.extend(semantic_expansions)
                
            except Exception as e:
                logging.warning(f"Semantic expansion failed: {e}")
        
        # Linguistic expansion
        linguistic_expansions = self.query_expander.expand_query_linguistic(normalized)
        expanded_terms.extend(linguistic_expansions)
        
        # Semantic expansion (if enabled)
        if include_semantic_expansion:
            try:
                semantic_expansions = await self.query_expander.expand_query_semantic(
                    normalized, max_expansions
                )
                expanded_terms.extend(semantic_expansions)
            except Exception as e:
                logging.warning(f"Semantic expansion failed: {e}")
        
        # Step 5: Rewrite query
        rewritten_variants = self.query_rewriter.rewrite_query(normalized, intent)
        
        # Calculate overall confidence
        confidence_factors = [intent_confidence]
        if entities:
            confidence_factors.append(0.8)  # Boost for entity presence
        if expanded_terms:
            confidence_factors.append(0.7)  # Boost for successful expansion
        
        overall_confidence = np.mean(confidence_factors)
        
        processing_time = time.time() - start_time
        
        # Update metrics
        self.enhancement_count += 1
        self.total_enhancement_time += processing_time
        
        # Create enhanced query
        enhanced = EnhancedQuery(
            original=query,
            normalized=normalized,
            expanded_terms=list(set(expanded_terms)),  # Remove duplicates
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
                'entity_count': len(entities)
            }
        )
        
        logging.debug(f"Enhanced query in {processing_time:.3f}s: {intent.value} intent, "
                     f"{len(expanded_terms)} expansions, {len(rewritten_variants)} rewrites")
        
        return enhanced
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_enhancement_time = self.total_enhancement_time / max(self.enhancement_count, 1)
        
        return {
            'total_enhancements': self.enhancement_count,
            'total_enhancement_time': self.total_enhancement_time,
            'avg_enhancement_time': avg_enhancement_time,
            'vocabulary_size': len(self.query_expander.vocabulary),
            'corpus_nodes': len(self.corpus_nodes)
        }
    
    def update_corpus(self, nodes: List[Node]):
        """Update corpus for expansion"""
        self.corpus_nodes = nodes
        self.query_expander = QueryExpander(self.embedding_model, nodes)
        logging.info(f"Updated corpus with {len(nodes)} nodes")


# Utility functions
def create_query_enhancer(embedding_model: BaseEmbeddingModel = None,
                         corpus_nodes: List[Node] = None) -> QueryEnhancer:
    """
    Create a query enhancer with specified configuration
    
    Args:
        embedding_model: Embedding model for semantic expansion
        corpus_nodes: Corpus nodes for vocabulary building
        
    Returns:
        Configured QueryEnhancer instance
    """
    return QueryEnhancer(embedding_model, corpus_nodes)


async def enhance_query_simple(query: str, enhancer: QueryEnhancer = None) -> EnhancedQuery:
    """
    Simple query enhancement function
    
    Args:
        query: Query to enhance
        enhancer: QueryEnhancer instance (created if None)
        
    Returns:
        Enhanced query
    """
    if enhancer is None:
        enhancer = create_query_enhancer()
    
    return await enhancer.enhance_query(query)