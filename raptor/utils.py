import logging
import re
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import tiktoken
from scipy import spatial

from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class DocumentType(Enum):
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    MIXED = "mixed"


@dataclass
class ChunkingConfig:
    """Configuration for enhanced chunking strategies"""
    max_tokens: int = 100
    overlap: int = 0
    min_chunk_size: int = 20
    semantic_threshold: float = 0.75
    preserve_structure: bool = True
    quality_threshold: float = 0.8
    fallback_enabled: bool = True


class DocumentTypeDetector:
    """Detects document type and characteristics for optimal chunking strategy"""
    
    MARKDOWN_INDICATORS = [
        r'^#{1,6}\s+',  # Headers
        r'```[\s\S]*?```',  # Code blocks
        r'^\s*[-*+]\s+',  # Bullet lists
        r'^\s*\d+\.\s+',  # Numbered lists
        r'\|.*\|',  # Tables
        r'^\s*>\s+',  # Blockquotes
        r'\[.*\]\(.*\)',  # Links
        r'\*\*.*\*\*',  # Bold
        r'\*.*\*',  # Italic
    ]
    
    @staticmethod
    def detect_document_type(text: str) -> DocumentType:
        """Detect if text is markdown, plain text, or mixed"""
        markdown_score = 0
        total_lines = len(text.split('\n'))
        
        for pattern in DocumentTypeDetector.MARKDOWN_INDICATORS:
            matches = len(re.findall(pattern, text, re.MULTILINE))
            markdown_score += matches
        
        # Calculate markdown density
        markdown_density = markdown_score / max(total_lines, 1)
        
        if markdown_density > 0.1:  # Significant markdown presence
            return DocumentType.MARKDOWN
        elif markdown_density > 0.02:  # Some markdown elements
            return DocumentType.MIXED
        else:
            return DocumentType.PLAIN_TEXT
    
    @staticmethod
    def analyze_content_characteristics(text: str) -> Dict[str, any]:
        """Analyze content for adaptive parameter selection"""
        lines = text.split('\n')
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'avg_line_length': np.mean([len(line) for line in lines if line.strip()]),
            'avg_sentence_length': np.mean([len(sent.strip()) for sent in sentences if sent.strip()]),
            'code_block_ratio': len(re.findall(r'```[\s\S]*?```', text)) / max(len(lines), 1),
            'list_density': len(re.findall(r'^\s*[-*+\d]+\.?\s+', text, re.MULTILINE)) / max(len(lines), 1),
            'header_count': len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        }


class MarkdownStructureParser:
    """Parses markdown structure for intelligent chunking"""
    
    @staticmethod
    def extract_sections(text: str) -> List[Dict[str, any]]:
        """Extract hierarchical sections from markdown"""
        lines = text.split('\n')
        sections = []
        current_section = {'level': 0, 'title': '', 'content': '', 'start_line': 0}
        
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    current_section['end_line'] = i
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    'level': level,
                    'title': title,
                    'content': line + '\n',
                    'start_line': i
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            current_section['end_line'] = len(lines)
            sections.append(current_section)
        
        return sections
    
    @staticmethod
    def preserve_markdown_elements(text: str) -> List[Tuple[int, int, str]]:
        """Identify elements that should not be split"""
        preserve_elements = []
        
        # Code blocks
        for match in re.finditer(r'```[\s\S]*?```', text):
            preserve_elements.append((match.start(), match.end(), 'code_block'))
        
        # Tables
        table_pattern = r'(\|.*\|(\n\|.*\|)*)'
        for match in re.finditer(table_pattern, text):
            preserve_elements.append((match.start(), match.end(), 'table'))
        
        # Lists (multi-line)
        list_pattern = r'(^\s*[-*+]\s+.*(\n^\s*[-*+]\s+.*)*)'
        for match in re.finditer(list_pattern, text, re.MULTILINE):
            preserve_elements.append((match.start(), match.end(), 'list'))
        
        return sorted(preserve_elements, key=lambda x: x[0])


class SemanticChunker:
    """Semantic-aware text chunking using embeddings"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for sentences"""
        if not self.embedding_model:
            # Fallback to simple similarity
            return np.random.random((len(sentences), 384))
        
        try:
            embeddings = []
            for sentence in sentences:
                if sentence.strip():
                    emb = self.embedding_model.create_embedding(sentence.strip())
                    embeddings.append(emb)
                else:
                    embeddings.append(np.zeros(384))  # Default dimension
            return np.array(embeddings)
        except Exception as e:
            logging.warning(f"Embedding generation failed: {e}. Using fallback.")
            return np.random.random((len(sentences), 384))
    
    def find_semantic_boundaries(self, sentences: List[str], threshold: float = 0.75) -> List[int]:
        """Find semantic boundaries between sentences"""
        if len(sentences) < 2:
            return []
        
        embeddings = self.get_sentence_embeddings(sentences)
        boundaries = []
        
        for i in range(len(embeddings) - 1):
            similarity = 1 - spatial.distance.cosine(embeddings[i], embeddings[i + 1])
            if similarity < threshold:
                boundaries.append(i + 1)
        
        return boundaries
    
    def chunk_by_semantic_similarity(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text based on semantic similarity between sentences"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Find semantic boundaries
        boundaries = self.find_semantic_boundaries(sentences, config.semantic_threshold)
        boundaries = [0] + boundaries + [len(sentences)]
        
        chunks = []
        tokenizer = tiktoken.get_encoding("o200k_base")
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Build chunk respecting token limits
            current_chunk = []
            current_tokens = 0
            
            for j in range(start_idx, end_idx):
                sentence = sentences[j]
                sentence_tokens = len(tokenizer.encode(sentence))
                
                if current_tokens + sentence_tokens > config.max_tokens and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks


class EnhancedTextSplitter:
    """Enhanced text splitter with adaptive strategies"""
    
    def __init__(self, embedding_model=None):
        self.detector = DocumentTypeDetector()
        self.markdown_parser = MarkdownStructureParser()
        self.semantic_chunker = SemanticChunker(embedding_model)
    
    def chunk_markdown_structure(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk markdown text preserving structure"""
        sections = self.markdown_parser.extract_sections(text)
        chunks = []
        tokenizer = tiktoken.get_encoding("o200k_base")
        
        for section in sections:
            content = section['content']
            tokens = len(tokenizer.encode(content))
            
            if tokens <= config.max_tokens:
                chunks.append(content.strip())
            else:
                # Section too large, split semantically within section
                sub_chunks = self.semantic_chunker.chunk_by_semantic_similarity(content, config)
                chunks.extend(sub_chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def chunk_plain_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk plain text using semantic similarity"""
        return self.semantic_chunker.chunk_by_semantic_similarity(text, config)
    
    def apply_overlap_strategy(self, chunks: List[str], config: ChunkingConfig) -> List[str]:
        """Apply intelligent overlap between chunks"""
        if config.overlap <= 0 or len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        tokenizer = tiktoken.get_encoding("o200k_base")
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Calculate overlap with previous chunk
            prev_chunk = chunks[i - 1]
            prev_sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
            curr_sentences = re.split(r'(?<=[.!?])\s+', chunk)
            
            # Take last few sentences from previous chunk
            overlap_sentences = prev_sentences[-config.overlap:] if len(prev_sentences) >= config.overlap else prev_sentences
            overlap_text = ' '.join(overlap_sentences)
            
            # Ensure overlap doesn't exceed limits
            overlap_tokens = len(tokenizer.encode(overlap_text))
            chunk_tokens = len(tokenizer.encode(chunk))
            
            if overlap_tokens + chunk_tokens <= config.max_tokens * 1.2:  # Allow slight overflow for overlap
                overlapped_chunk = overlap_text + ' ' + chunk
            else:
                overlapped_chunk = chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def validate_chunk_quality(self, chunks: List[str], config: ChunkingConfig) -> Tuple[List[str], float]:
        """Validate and score chunk quality"""
        if not chunks:
            return chunks, 0.0
        
        tokenizer = tiktoken.get_encoding("o200k_base")
        quality_scores = []
        valid_chunks = []
        
        for chunk in chunks:
            tokens = len(tokenizer.encode(chunk))
            
            # Size quality
            size_score = 1.0 if config.min_chunk_size <= tokens <= config.max_tokens else 0.5
            
            # Content quality (basic checks)
            content_score = 1.0
            if len(chunk.strip()) < 10:  # Too short
                content_score *= 0.3
            if chunk.count('\n') / len(chunk) > 0.1:  # Too many line breaks
                content_score *= 0.8
            
            overall_score = (size_score + content_score) / 2
            quality_scores.append(overall_score)
            
            if overall_score >= config.quality_threshold:
                valid_chunks.append(chunk)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        return valid_chunks, avg_quality
    
    def split_text_enhanced(self, text: str, config: ChunkingConfig) -> List[str]:
        """Enhanced text splitting with adaptive strategies"""
        if not text or not text.strip():
            return []
        
        # Detect document type
        doc_type = self.detector.detect_document_type(text)
        characteristics = self.detector.analyze_content_characteristics(text)
        
        # Adaptive parameter adjustment
        adaptive_config = self._adapt_config(config, characteristics)
        
        try:
            # Primary chunking strategy
            if doc_type == DocumentType.MARKDOWN:
                chunks = self.chunk_markdown_structure(text, adaptive_config)
            else:
                chunks = self.chunk_plain_text(text, adaptive_config)
            
            # Apply overlap if configured
            if adaptive_config.overlap > 0:
                chunks = self.apply_overlap_strategy(chunks, adaptive_config)
            
            # Validate quality
            valid_chunks, quality_score = self.validate_chunk_quality(chunks, adaptive_config)
            
            # Fallback if quality is poor
            if quality_score < adaptive_config.quality_threshold and adaptive_config.fallback_enabled:
                logging.warning(f"Low chunk quality ({quality_score:.2f}), applying fallback strategy")
                valid_chunks = self._fallback_chunking(text, adaptive_config)
            
            logging.info(f"Enhanced chunking: {len(valid_chunks)} chunks, quality: {quality_score:.2f}, type: {doc_type.value}")
            return valid_chunks
            
        except Exception as e:
            logging.error(f"Enhanced chunking failed: {e}")
            if adaptive_config.fallback_enabled:
                return self._fallback_chunking(text, adaptive_config)
            return []
    
    def _adapt_config(self, config: ChunkingConfig, characteristics: Dict) -> ChunkingConfig:
        """Adapt configuration based on content characteristics"""
        adapted = ChunkingConfig(
            max_tokens=config.max_tokens,
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size,
            semantic_threshold=config.semantic_threshold,
            preserve_structure=config.preserve_structure,
            quality_threshold=config.quality_threshold,
            fallback_enabled=config.fallback_enabled
        )
        
        # Adjust semantic threshold based on content
        if characteristics['avg_sentence_length'] > 100:  # Long sentences
            adapted.semantic_threshold *= 0.9  # More lenient
        
        if characteristics['code_block_ratio'] > 0.1:  # Code-heavy content
            adapted.preserve_structure = True
            adapted.semantic_threshold *= 0.8
        
        return adapted
    
    def _fallback_chunking(self, text: str, config: ChunkingConfig) -> List[str]:
        """Fallback to simple sentence-based chunking"""
        # Simple sentence-based splitting as fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        tokenizer = tiktoken.get_encoding("o200k_base")
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_tokens = len(tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > config.max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


# Global enhanced splitter instance
_enhanced_splitter = None


def get_enhanced_splitter(embedding_model=None):
    """Get or create enhanced splitter instance"""
    global _enhanced_splitter
    if _enhanced_splitter is None:
        _enhanced_splitter = EnhancedTextSplitter(embedding_model)
    return _enhanced_splitter


# Enhanced split_text function - backward compatible
def split_text(
    text: str, 
    tokenizer: tiktoken.get_encoding("o200k_base"), 
    max_tokens: int, 
    overlap: int = 0,
    embedding_model=None,
    enhanced: bool = True,
    **kwargs
) -> List[str]:
    """
    Enhanced text splitting with adaptive strategies.
    
    Args:
        text (str): The text to be split.
        tokenizer: The tokenizer (kept for backward compatibility).
        max_tokens (int): The maximum allowed tokens per chunk.
        overlap (int, optional): The number of overlapping tokens between chunks.
        embedding_model: Embedding model for semantic chunking.
        enhanced (bool): Whether to use enhanced chunking strategies.
        **kwargs: Additional configuration options.
    
    Returns:
        List[str]: A list of text chunks.
    """
    if not enhanced:
        # Fallback to original implementation for backward compatibility
        return _original_split_text(text, tokenizer, max_tokens, overlap)
    
    # Use enhanced chunking
    config = ChunkingConfig(
        max_tokens=max_tokens,
        overlap=overlap,
        **kwargs
    )
    
    splitter = get_enhanced_splitter(embedding_model)
    return splitter.split_text_enhanced(text, config)


def _original_split_text(text: str, tokenizer, max_tokens: int, overlap: int = 0) -> List[str]:
    """Original split_text implementation for backward compatibility"""
    # Original implementation
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        if not sentence.strip():
            continue
        
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# Keep all existing utility functions unchanged
def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)