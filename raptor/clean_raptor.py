# clean_raptor_system.py - TEMİZ VE HIZLI RAPTOR SİSTEMİ
import logging
import time
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import pickle
import tiktoken
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Temel bağımlılıklar
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ============================================================================
# TEMEİ YAPILAR - Karmaşık inheritance yok
# ============================================================================

@dataclass
class Node:
    """Basit node yapısı"""
    text: str
    index: int
    children: Set[int]
    embedding: np.ndarray
    layer: int = 0

@dataclass
class Tree:
    """Basit tree yapısı"""
    nodes: Dict[int, Node]
    layers: Dict[int, List[Node]]
    root_nodes: List[Node]
    leaf_nodes: List[Node]

# ============================================================================
# TEMİZ MODEL WRAPPER'LARI - Minimal interface
# ============================================================================

class SimpleEmbeddingModel:
    """Temiz embedding model - gereksiz cache yok"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logging.info(f"Embedding model loaded: {model_name}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Batch encoding - basit ve hızlı"""
        return self.model.encode(texts, show_progress_bar=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Tek text encoding"""
        return self.model.encode([text])[0]

class SimpleSummarizationModel:
    """Temiz summarization model"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        logging.info(f"Summarization model: {model}")
    
    def summarize(self, text: str, max_tokens: int = 200) -> str:
        """Basit summarization - gereksiz retry yok"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": f"Summarize this text in {max_tokens} tokens or less:\n\n{text}"}
                ],
                max_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return text[:1000]  # Fallback

class SimpleQAModel:
    """Temiz QA model"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
    
    def answer(self, context: str, question: str) -> str:
        """Basit QA - gereksiz wrapper yok"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"QA failed: {e}")
            return "Unable to answer the question."

# ============================================================================
# TEMİZ UTILITIES - Gereksiz abstraction yok
# ============================================================================

def simple_text_splitter(text: str, max_tokens: int = 100) -> List[str]:
    """Basit text splitter - gereksiz enhancement yok"""
    tokenizer = tiktoken.get_encoding("o200k_base")
    
    # Sentence'lara böl
    sentences = text.replace('\n', ' ').split('. ')
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence = sentence.strip() + '.'
        tokens = len(tokenizer.encode(sentence))
        
        if current_tokens + tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = tokens
        else:
            current_chunk.append(sentence)
            current_tokens += tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def simple_clustering(embeddings: np.ndarray, max_clusters: int = 5) -> List[List[int]]:
    """Basit clustering - gereksiz adaptive parametreler yok"""
    if len(embeddings) <= 2:
        return [list(range(len(embeddings)))]
    
    n_clusters = min(max_clusters, len(embeddings) // 2)
    n_clusters = max(2, n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    clusters = []
    for i in range(n_clusters):
        cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
        if cluster_indices:
            clusters.append(cluster_indices)
    
    return clusters

# ============================================================================
# TEMİZ RAPTOR BUILDER - Karmaşık inheritance yok
# ============================================================================

class CleanRaptorBuilder:
    """Temiz RAPTOR builder - gereksiz optimization wrapper'ları yok"""
    
    def __init__(self, 
                 embedding_model: SimpleEmbeddingModel,
                 summarization_model: SimpleSummarizationModel):
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.tokenizer = tiktoken.get_encoding("o200k_base")
    
    def build_tree(self, text: str, max_tokens: int = 100, max_layers: int = 3) -> Tree:
        """Ana tree building methodu - temiz ve basit"""
        logging.info(f"Building RAPTOR tree from {len(text)} characters")
        
        # 1. Text'i chunk'lara böl
        chunks = simple_text_splitter(text, max_tokens)
        logging.info(f"Created {len(chunks)} chunks")
        
        # 2. Leaf node'ları oluştur
        leaf_embeddings = self.embedding_model.encode(chunks)
        
        leaf_nodes = []
        for i, (chunk, embedding) in enumerate(zip(chunks, leaf_embeddings)):
            node = Node(
                text=chunk,
                index=i,
                children=set(),
                embedding=embedding,
                layer=0
            )
            leaf_nodes.append(node)
        
        # 3. Layer'ları oluştur
        all_nodes = {node.index: node for node in leaf_nodes}
        layers = {0: leaf_nodes}
        current_nodes = leaf_nodes
        next_index = len(leaf_nodes)
        layer = 0
        
        while len(current_nodes) > 1 and layer < max_layers:
            layer += 1
            logging.info(f"Building layer {layer} from {len(current_nodes)} nodes")
            
            # Embeddings'leri al
            embeddings = np.array([node.embedding for node in current_nodes])
            
            # Cluster yap
            clusters = simple_clustering(embeddings)
            
            if len(clusters) >= len(current_nodes):
                logging.info("No reduction in clustering, stopping")
                break
            
            # Her cluster için parent node oluştur
            new_nodes = []
            for cluster_indices in clusters:
                cluster_nodes = [current_nodes[i] for i in cluster_indices]
                
                # Cluster text'ini birleştir
                cluster_text = "\n\n".join([node.text for node in cluster_nodes])
                
                # Summarize et
                summary = self.summarization_model.summarize(cluster_text)
                
                # Embedding oluştur
                summary_embedding = self.embedding_model.encode_single(summary)
                
                # Parent node oluştur
                parent_node = Node(
                    text=summary,
                    index=next_index,
                    children={node.index for node in cluster_nodes},
                    embedding=summary_embedding,
                    layer=layer
                )
                
                new_nodes.append(parent_node)
                all_nodes[next_index] = parent_node
                next_index += 1
            
            layers[layer] = new_nodes
            current_nodes = new_nodes
            
            logging.info(f"Layer {layer}: {len(new_nodes)} nodes created")
        
        # Tree'yi oluştur
        tree = Tree(
            nodes=all_nodes,
            layers=layers,
            root_nodes=current_nodes,
            leaf_nodes=leaf_nodes
        )
        
        logging.info(f"Tree built: {len(layers)} layers, {len(all_nodes)} total nodes")
        return tree

# ============================================================================
# TEMİZ RETRIEVER - Gereksiz cache ve fusion yok
# ============================================================================

class CleanRetriever:
    """Temiz retriever - karmaşık hybrid system yok"""
    
    def __init__(self, tree: Tree, embedding_model: SimpleEmbeddingModel):
        self.tree = tree
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        
        # Opsiyonel sparse retrieval için
        all_texts = [node.text for node in tree.nodes.values()]
        tokenized_texts = [text.lower().split() for text in all_texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        self.all_nodes = list(tree.nodes.values())
    
    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Node]:
        """Dense retrieval - basit cosine similarity"""
        query_embedding = self.embedding_model.encode_single(query)
        
        # Tüm node'larla similarity hesapla
        similarities = []
        for node in self.tree.nodes.values():
            sim = cosine_similarity([query_embedding], [node.embedding])[0][0]
            similarities.append((sim, node))
        
        # En yüksek similarity'li node'ları döndür
        similarities.sort(reverse=True)
        return [node for _, node in similarities[:top_k]]
    
    def retrieve_sparse(self, query: str, top_k: int = 5) -> List[Node]:
        """Sparse retrieval - basit BM25"""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # En yüksek score'lu node'ları döndür
        scored_nodes = [(score, node) for score, node in zip(scores, self.all_nodes)]
        scored_nodes.sort(reverse=True)
        
        return [node for _, node in scored_nodes[:top_k]]
    
    def retrieve_hybrid(self, query: str, top_k: int = 5, dense_weight: float = 0.7) -> List[Node]:
        """Basit hybrid retrieval - gereksiz fusion complexity yok"""
        dense_nodes = self.retrieve_dense(query, top_k * 2)
        sparse_nodes = self.retrieve_sparse(query, top_k * 2)
        
        # Basit scoring
        node_scores = {}
        
        # Dense scores
        for i, node in enumerate(dense_nodes):
            score = (len(dense_nodes) - i) * dense_weight
            node_scores[node.index] = node_scores.get(node.index, 0) + score
        
        # Sparse scores
        sparse_weight = 1.0 - dense_weight
        for i, node in enumerate(sparse_nodes):
            score = (len(sparse_nodes) - i) * sparse_weight
            node_scores[node.index] = node_scores.get(node.index, 0) + score
        
        # En yüksek score'lu node'ları döndür
        sorted_nodes = sorted(
            [(score, self.tree.nodes[node_id]) for node_id, score in node_scores.items()],
            reverse=True
        )
        
        return [node for _, node in sorted_nodes[:top_k]]
    
    def get_context(self, nodes: List[Node], max_tokens: int = 3000) -> str:
        """Node'lardan context oluştur"""
        context_parts = []
        total_tokens = 0
        
        for node in nodes:
            node_tokens = len(self.tokenizer.encode(node.text))
            if total_tokens + node_tokens > max_tokens:
                break
            
            context_parts.append(node.text)
            total_tokens += node_tokens
        
        return "\n\n".join(context_parts)

# ============================================================================
# TEMİZ RAPTOR SİSTEMİ - Tek sınıf, basit API
# ============================================================================

class CleanRAPTOR:
    """Temiz RAPTOR sistemi - gereksiz wrapper'lar yok"""
    
    def __init__(self,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 summarization_model: str = "gpt-4o-mini",
                 qa_model: str = "gpt-4o-mini"):
        
        logging.info("Initializing Clean RAPTOR system")
        
        self.embedding_model = SimpleEmbeddingModel(embedding_model_name)
        self.summarization_model = SimpleSummarizationModel(summarization_model)
        self.qa_model = SimpleQAModel(qa_model)
        
        self.tree = None
        self.retriever = None
    
    def add_documents(self, text: str, max_tokens: int = 100, max_layers: int = 3):
        """Doküman ekle ve tree oluştur"""
        builder = CleanRaptorBuilder(self.embedding_model, self.summarization_model)
        self.tree = builder.build_tree(text, max_tokens, max_layers)
        self.retriever = CleanRetriever(self.tree, self.embedding_model)
        logging.info("Documents added and tree built")
    
    def retrieve(self, query: str, method: str = "hybrid", top_k: int = 5, max_tokens: int = 3000) -> str:
        """Basit retrieval"""
        if not self.retriever:
            raise ValueError("No documents added yet")
        
        if method == "dense":
            nodes = self.retriever.retrieve_dense(query, top_k)
        elif method == "sparse":
            nodes = self.retriever.retrieve_sparse(query, top_k)
        else:  # hybrid
            nodes = self.retriever.retrieve_hybrid(query, top_k)
        
        return self.retriever.get_context(nodes, max_tokens)
    
    def answer_question(self, question: str, method: str = "hybrid", top_k: int = 5) -> str:
        """Soru cevaplama"""
        context = self.retrieve(question, method, top_k)
        return self.qa_model.answer(context, question)
    
    def save(self, path: str):
        """Tree'yi kaydet"""
        if not self.tree:
            raise ValueError("No tree to save")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.tree, f)
        
        logging.info(f"Tree saved to {path}")
    
    def load(self, path: str):
        """Tree'yi yükle"""
        with open(path, 'rb') as f:
            self.tree = pickle.load(f)
        
        self.retriever = CleanRetriever(self.tree, self.embedding_model)
        logging.info(f"Tree loaded from {path}")
    
    def get_stats(self) -> Dict:
        """Basit istatistikler"""
        if not self.tree:
            return {"status": "No tree built"}
        
        return {
            "total_nodes": len(self.tree.nodes),
            "layers": len(self.tree.layers),
            "leaf_nodes": len(self.tree.leaf_nodes),
            "root_nodes": len(self.tree.root_nodes)
        }

# ============================================================================
# KULLANIM ÖRNEĞİ
# ============================================================================

def main():
    """Temiz RAPTOR kullanım örneği"""
    
    # Sistem oluştur
    raptor = CleanRAPTOR()
    
    # Test metni
    text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
    Machine learning is a subset of AI that focuses on algorithms that can learn from data.
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    Natural language processing (NLP) is another branch of AI that deals with human language.
    Computer vision is the field of AI that enables machines to interpret visual information.
    """
    
    # Tree oluştur
    print("Building tree...")
    start_time = time.time()
    raptor.add_documents(text)
    build_time = time.time() - start_time
    print(f"Tree built in {build_time:.2f} seconds")
    
    # İstatistikleri göster
    stats = raptor.get_stats()
    print(f"Stats: {stats}")
    
    # Test sorular
    queries = [
        "What is artificial intelligence?",
        "Tell me about machine learning",
        "What is deep learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Retrieval test
        start_time = time.time()
        context = raptor.retrieve(query, method="hybrid", top_k=3)
        retrieval_time = time.time() - start_time
        print(f"Retrieved in {retrieval_time:.3f}s: {context[:200]}...")
        
        # QA test
        start_time = time.time()
        answer = raptor.answer_question(query)
        qa_time = time.time() - start_time
        print(f"Answered in {qa_time:.3f}s: {answer}")

if __name__ == "__main__":
    main()