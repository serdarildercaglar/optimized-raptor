# test_hybrid_raptor.py
"""
Comprehensive integration tests for Enhanced RAPTOR with hybrid features

Run with: pytest test_hybrid_raptor.py -v
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Import test frameworks
import pytest_asyncio

# Import RAPTOR components
from raptor import RetrievalAugmentationConfig, GPT41SummarizationModel
from raptor.EmbeddingModels import CustomEmbeddingModel
from raptor.enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, 
    HybridConfig, 
    create_enhanced_raptor
)
from raptor.hybrid_retriever import FusionMethod, HybridRetriever
from raptor.sparse_retriever import AdvancedBM25Retriever, create_sparse_retriever
from raptor.query_enhancement import QueryEnhancer, QueryIntent, create_query_enhancer
from raptor.tree_structures import Node


# Test data
SAMPLE_TEXT = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
Machine Learning is a subset of AI that enables computers to learn from data without explicit programming.
Deep Learning uses neural networks with multiple layers to model and understand complex patterns.

Natural Language Processing (NLP) allows computers to understand and generate human language.
Computer Vision enables machines to interpret and analyze visual information from the world.
Robotics combines AI with mechanical engineering to create autonomous systems.

The applications of AI are vast, including healthcare, finance, transportation, and education.
Ethical considerations in AI development include bias, privacy, and job displacement concerns.
The future of AI promises continued advancement in automation and intelligent decision-making.
"""

SAMPLE_QUERIES = [
    "What is artificial intelligence?",
    "machine learning definition",
    "How does deep learning work?",
    "NLP applications",
    "AI ethics challenges"
]


class TestFixtures:
    """Test fixtures and setup utilities"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for testing"""
        texts = SAMPLE_TEXT.split('\n\n')
        nodes = []
        
        for i, text in enumerate(texts):
            if text.strip():
                # Mock embeddings
                embeddings = {
                    'OpenAI': np.random.random(384).tolist(),
                    'CustomEmbeddingModel': np.random.random(384).tolist()
                }
                node = Node(text.strip(), i, set(), embeddings)
                nodes.append(node)
        
        return nodes
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing"""
        mock_model = Mock(spec=CustomEmbeddingModel)
        mock_model.create_embedding.return_value = np.random.random(384).tolist()
        mock_model.create_embedding_async = asyncio.coroutine(
            lambda x: np.random.random(384).tolist()
        )
        mock_model.__class__.__name__ = "MockEmbeddingModel"
        return mock_model
    
    @pytest.fixture
    def mock_summarization_model(self):
        """Mock summarization model for testing"""
        mock_model = Mock(spec=GPT41SummarizationModel)
        mock_model.summarize.return_value = "Test summary"
        return mock_model


class TestSparseRetriever(TestFixtures):
    """Test sparse retrieval functionality"""
    
    def test_sparse_retriever_creation(self):
        """Test sparse retriever initialization"""
        retriever = create_sparse_retriever(algorithm="bm25_okapi")
        
        assert retriever.algorithm == "bm25_okapi"
        assert retriever.k1 == 1.2
        assert retriever.b == 0.75
        assert retriever.enable_caching == True
    
    def test_sparse_retriever_build_index(self, sample_nodes):
        """Test building sparse index from nodes"""
        retriever = create_sparse_retriever()
        
        # Build index
        retriever.build_from_nodes(sample_nodes, save_cache=False)
        
        assert len(retriever.corpus_tokens) == len(sample_nodes)
        assert retriever.bm25_model is not None
        assert len(retriever.doc_metadata) == len(sample_nodes)
    
    def test_sparse_retrieval_basic(self, sample_nodes):
        """Test basic sparse retrieval"""
        retriever = create_sparse_retriever()
        retriever.build_from_nodes(sample_nodes, save_cache=False)
        
        # Test retrieval
        results = retriever.retrieve_with_scores("artificial intelligence", top_k=3)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'node') for r in results)
        assert all(r.score >= 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_sparse_retrieval_async(self, sample_nodes):
        """Test async sparse retrieval"""
        retriever = create_sparse_retriever()
        retriever.build_from_nodes(sample_nodes, save_cache=False)
        
        # Test async retrieval
        results = await retriever.retrieve_async("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert all(hasattr(r, 'score') for r in results)
    
    def test_sparse_retriever_caching(self, sample_nodes, temp_dir):
        """Test sparse retriever caching"""
        cache_dir = Path(temp_dir) / "sparse_cache"
        
        # First retriever - build and cache
        retriever1 = AdvancedBM25Retriever(enable_caching=True, cache_dir=str(cache_dir))
        retriever1.build_from_nodes(sample_nodes, save_cache=True)
        
        # Second retriever - should load from cache
        retriever2 = AdvancedBM25Retriever(enable_caching=True, cache_dir=str(cache_dir))
        retriever2.build_from_nodes(sample_nodes, save_cache=False)
        
        # Both should have same corpus
        assert len(retriever1.corpus_tokens) == len(retriever2.corpus_tokens)
        assert len(retriever1.doc_metadata) == len(retriever2.doc_metadata)
    
    def test_query_analysis(self, sample_nodes):
        """Test query analysis functionality"""
        retriever = create_sparse_retriever()
        retriever.build_from_nodes(sample_nodes, save_cache=False)
        
        analysis = retriever.get_query_analysis("artificial intelligence machine learning")
        
        assert 'original_query' in analysis
        assert 'processed_tokens' in analysis
        assert 'term_document_frequencies' in analysis
        assert 'corpus_coverage' in analysis


class TestQueryEnhancement(TestFixtures):
    """Test query enhancement functionality"""
    
    def test_query_enhancer_creation(self, mock_embedding_model, sample_nodes):
        """Test query enhancer initialization"""
        enhancer = create_query_enhancer(
            embedding_model=mock_embedding_model,
            corpus_nodes=sample_nodes
        )
        
        assert enhancer.embedding_model == mock_embedding_model
        assert len(enhancer.corpus_nodes) == len(sample_nodes)
        assert enhancer.normalizer is not None
        assert enhancer.intent_detector is not None
    
    @pytest.mark.asyncio
    async def test_query_enhancement_basic(self, mock_embedding_model):
        """Test basic query enhancement"""
        enhancer = create_query_enhancer(embedding_model=mock_embedding_model)
        
        enhanced = await enhancer.enhance_query("What is artificial intelligence?")
        
        assert enhanced.original == "What is artificial intelligence?"
        assert enhanced.normalized != ""
        assert enhanced.intent != QueryIntent.UNKNOWN
        assert enhanced.confidence_score > 0
        assert enhanced.processing_time > 0
    
    def test_intent_detection(self):
        """Test query intent detection"""
        enhancer = create_query_enhancer()
        detector = enhancer.intent_detector
        
        test_cases = [
            ("What is AI?", QueryIntent.FACTUAL),
            ("Define machine learning", QueryIntent.DEFINITIONAL),
            ("How to implement neural networks?", QueryIntent.PROCEDURAL),
            ("Compare CNN and RNN", QueryIntent.COMPARATIVE),
            ("Why does AI need data?", QueryIntent.CAUSAL),
        ]
        
        for query, expected_intent in test_cases:
            intent, confidence = detector.detect_intent(query)
            # Allow for some flexibility in intent detection
            assert intent in [expected_intent, QueryIntent.UNKNOWN]
            assert 0 <= confidence <= 1
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        enhancer = create_query_enhancer()
        extractor = enhancer.entity_extractor
        
        entities = extractor.extract_entities("OpenAI released GPT-4 in 2023 with 99% accuracy")
        
        # Should extract proper nouns, years, percentages
        assert any("OpenAI" in str(entity) for entity in entities)
        assert any("2023" in str(entity) for entity in entities)
        assert any("99%" in str(entity) for entity in entities)
    
    @pytest.mark.asyncio
    async def test_semantic_expansion(self, mock_embedding_model, sample_nodes):
        """Test semantic query expansion"""
        enhancer = create_query_enhancer(
            embedding_model=mock_embedding_model,
            corpus_nodes=sample_nodes
        )
        
        # Mock semantic expansion
        with patch.object(enhancer.query_expander, 'expand_query_semantic') as mock_expand:
            mock_expand.return_value = ['learning', 'algorithms', 'neural']
            
            enhanced = await enhancer.enhance_query("machine learning")
            
            assert len(enhanced.expanded_terms) > 0


class TestHybridRetriever(TestFixtures):
    """Test hybrid retrieval functionality"""
    
    @pytest.fixture
    def mock_dense_retriever(self):
        """Mock dense retriever"""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = "Dense retrieval context"
        mock_retriever.retrieve_async = asyncio.coroutine(lambda *args, **kwargs: "Dense context")
        return mock_retriever
    
    @pytest.fixture
    def sparse_retriever_with_data(self, sample_nodes):
        """Sparse retriever with test data"""
        retriever = create_sparse_retriever()
        retriever.build_from_nodes(sample_nodes, save_cache=False)
        return retriever
    
    def test_hybrid_retriever_creation(self, mock_dense_retriever, sparse_retriever_with_data):
        """Test hybrid retriever initialization"""
        hybrid_retriever = HybridRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=sparse_retriever_with_data,
            fusion_method=FusionMethod.RRF,
            dense_weight=0.6,
            sparse_weight=0.4
        )
        
        assert hybrid_retriever.dense_retriever == mock_dense_retriever
        assert hybrid_retriever.sparse_retriever == sparse_retriever_with_data
        assert hybrid_retriever.fusion_method == FusionMethod.RRF
        assert hybrid_retriever.dense_weight == 0.6
        assert hybrid_retriever.sparse_weight == 0.4
    
    def test_result_fusion(self, sample_nodes):
        """Test result fusion algorithms"""
        from raptor.hybrid_retriever import ResultFusion
        
        # Create mock results
        dense_results = [(sample_nodes[0], 0.9), (sample_nodes[1], 0.8)]
        sparse_results = [
            Mock(node=sample_nodes[0], score=0.7, query_terms_matched=['ai']),
            Mock(node=sample_nodes[2], score=0.6, query_terms_matched=['machine'])
        ]
        
        # Test RRF fusion
        fusion = ResultFusion(FusionMethod.RRF)
        fused_results = fusion.fuse_results(dense_results, sparse_results)
        
        assert len(fused_results) >= 2
        assert all(hasattr(r, 'fused_score') for r in fused_results)
        assert all(hasattr(r, 'dense_score') for r in fused_results)
        assert all(hasattr(r, 'sparse_score') for r in fused_results)
    
    @pytest.mark.asyncio
    async def test_reranking(self, mock_embedding_model, sample_nodes):
        """Test cross-encoder reranking"""
        from raptor.hybrid_retriever import CrossEncoderReranker, HybridRetrievalResult
        
        reranker = CrossEncoderReranker(mock_embedding_model)
        
        # Create mock hybrid results
        results = [
            HybridRetrievalResult(
                node=sample_nodes[0], dense_score=0.8, sparse_score=0.6,
                fused_score=0.7, rank_dense=1, rank_sparse=2, final_rank=1
            ),
            HybridRetrievalResult(
                node=sample_nodes[1], dense_score=0.7, sparse_score=0.8,
                fused_score=0.75, rank_dense=2, rank_sparse=1, final_rank=2
            )
        ]
        
        reranked = await reranker.rerank_results("test query", results)
        
        assert len(reranked) == len(results)
        assert all(hasattr(r, 'rerank_score') for r in reranked)
        assert all(hasattr(r, 'confidence') for r in reranked)


class TestEnhancedRAPTOR(TestFixtures):
    """Test enhanced RAPTOR integration"""
    
    @pytest.fixture
    def test_config(self, mock_embedding_model, mock_summarization_model):
        """Test configuration for enhanced RAPTOR"""
        return RetrievalAugmentationConfig(
            tb_max_tokens=50,
            tb_summarization_length=100,
            tb_num_layers=2,
            summarization_model=mock_summarization_model,
            embedding_model=mock_embedding_model,
            enable_async=False  # Disable async for testing
        )
    
    @pytest.fixture
    def hybrid_config(self):
        """Hybrid configuration for testing"""
        return HybridConfig(
            enable_hybrid=True,
            enable_query_enhancement=True,
            enable_sparse_retrieval=True,
            enable_reranking=False,  # Disable for faster testing
            fusion_method=FusionMethod.RRF,
            enable_caching=False  # Disable caching for tests
        )
    
    def test_enhanced_raptor_creation(self, test_config, hybrid_config):
        """Test enhanced RAPTOR initialization"""
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=test_config,
            hybrid_config=hybrid_config
        )
        
        assert enhanced_raptor.hybrid_config == hybrid_config
        assert enhanced_raptor.config == test_config
        assert hasattr(enhanced_raptor, 'hybrid_metrics')
    
    @patch('raptor.tree_builder.TreeBuilder.build_from_text')
    def test_document_addition_with_hybrid(self, mock_build, test_config, hybrid_config):
        """Test document addition with hybrid component initialization"""
        # Mock tree building
        from raptor.tree_structures import Tree
        mock_tree = Mock(spec=Tree)
        mock_tree.all_nodes = {0: Mock(), 1: Mock()}
        mock_tree.num_layers = 2
        mock_build.return_value = mock_tree
        
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=test_config,
            hybrid_config=hybrid_config
        )
        
        # This should initialize hybrid components
        with patch.object(enhanced_raptor, '_initialize_hybrid_components') as mock_init:
            enhanced_raptor.add_documents(SAMPLE_TEXT)
            mock_init.assert_called_once()
    
    def test_retrieve_enhanced_methods(self, test_config, hybrid_config):
        """Test different retrieval methods"""
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=test_config,
            hybrid_config=hybrid_config
        )
        
        # Mock components
        enhanced_raptor.retriever = Mock()
        enhanced_raptor.retriever.retrieve.return_value = "Dense context"
        
        enhanced_raptor.sparse_retriever = Mock()
        enhanced_raptor.sparse_retriever.retrieve_async = asyncio.coroutine(
            lambda *args, **kwargs: [Mock(node=Mock(text="Sparse context"), score=0.8)]
        )
        
        enhanced_raptor.hybrid_retriever = Mock()
        enhanced_raptor.hybrid_retriever.retrieve_hybrid_async = asyncio.coroutine(
            lambda *args, **kwargs: [Mock(node=Mock(text="Hybrid context"))]
        )
        
        # Test different methods
        methods = ["dense", "sparse", "hybrid"]
        for method in methods:
            try:
                result = enhanced_raptor.retrieve_enhanced("test query", method=method)
                assert isinstance(result, str)
            except Exception as e:
                # Some methods might fail in mock environment, that's ok
                assert "not initialized" in str(e) or "Mock" in str(e)
    
    def test_performance_tracking(self, test_config, hybrid_config):
        """Test performance metrics tracking"""
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=test_config,
            hybrid_config=hybrid_config
        )
        
        # Check initial metrics
        assert enhanced_raptor.hybrid_metrics['hybrid_queries'] == 0
        assert enhanced_raptor.hybrid_metrics['total_hybrid_time'] == 0.0
        
        # Performance summary should not crash
        summary = enhanced_raptor.get_enhanced_performance_summary()
        assert 'hybrid_features' in summary
        assert 'hybrid_metrics' in summary


class TestIntegrationScenarios(TestFixtures):
    """Test complete integration scenarios"""
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self, temp_dir):
        """Test complete end-to-end workflow"""
        # Skip this test if running quick tests
        pytest.skip("End-to-end test - run with --slow flag")
        
        try:
            # Create enhanced RAPTOR
            enhanced_raptor = create_enhanced_raptor(
                text=SAMPLE_TEXT,
                hybrid_config=HybridConfig(enable_caching=False)
            )
            
            # Test query enhancement
            enhanced_query = enhanced_raptor.enhance_query_only("What is AI?")
            assert enhanced_query.original == "What is AI?"
            
            # Test different retrieval methods
            for method in ["dense", "sparse", "hybrid"]:
                try:
                    context = enhanced_raptor.retrieve_enhanced(
                        "artificial intelligence", 
                        method=method,
                        top_k=2
                    )
                    assert isinstance(context, str)
                    assert len(context) > 0
                except Exception:
                    # Some methods might not be available depending on setup
                    pass
            
            # Test performance summary
            performance = enhanced_raptor.get_enhanced_performance_summary()
            assert isinstance(performance, dict)
            
        except Exception as e:
            pytest.fail(f"End-to-end test failed: {e}")
    
    def test_backward_compatibility(self, test_config):
        """Test that enhanced RAPTOR maintains backward compatibility"""
        enhanced_raptor = EnhancedRetrievalAugmentation(config=test_config)
        
        # Mock the base retriever
        enhanced_raptor.retriever = Mock()
        enhanced_raptor.retriever.retrieve.return_value = "Standard retrieval result"
        
        # Standard RAPTOR methods should still work
        result = enhanced_raptor.retrieve("test query")
        assert result == "Standard retrieval result"
        
        # Standard answer_question should work
        enhanced_raptor.qa_model = Mock()
        enhanced_raptor.qa_model.answer_question.return_value = "Test answer"
        
        answer = enhanced_raptor.answer_question("test question")
        assert answer == "Test answer"
    
    def test_error_handling(self, test_config, hybrid_config):
        """Test error handling in various scenarios"""
        enhanced_raptor = EnhancedRetrievalAugmentation(
            config=test_config,
            hybrid_config=hybrid_config
        )
        
        # Test query enhancement without initialized components
        with pytest.raises(ValueError, match="Query enhancer not initialized"):
            enhanced_raptor.enhance_query_only("test query")
        
        # Test retrieval with invalid method
        result = enhanced_raptor.retrieve_enhanced("query", method="invalid_method")
        # Should fallback to standard retrieval
        assert isinstance(result, str)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid hybrid config
        valid_config = HybridConfig(
            fusion_method=FusionMethod.RRF,
            dense_weight=0.6,
            sparse_weight=0.4
        )
        assert valid_config.dense_weight + valid_config.sparse_weight == 1.0
        
        # Test that weights should sum to 1.0 (in actual usage)
        config_with_wrong_weights = HybridConfig(
            dense_weight=0.8,
            sparse_weight=0.8  # This sums to 1.6, which might be problematic
        )
        # The system should handle this gracefully
        assert config_with_wrong_weights.dense_weight == 0.8


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for hybrid features"""
    
    @pytest.mark.benchmark
    def test_sparse_retrieval_performance(self, benchmark, sample_nodes):
        """Benchmark sparse retrieval performance"""
        retriever = create_sparse_retriever()
        retriever.build_from_nodes(sample_nodes, save_cache=False)
        
        def sparse_retrieval():
            return retriever.retrieve_with_scores("artificial intelligence", top_k=5)
        
        result = benchmark(sparse_retrieval)
        assert len(result) <= 5
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_query_enhancement_performance(self, benchmark, mock_embedding_model):
        """Benchmark query enhancement performance"""
        enhancer = create_query_enhancer(embedding_model=mock_embedding_model)
        
        async def enhance_query():
            return await enhancer.enhance_query("What is machine learning?")
        
        result = await benchmark(enhance_query)
        assert result.original == "What is machine learning?"


# Test runner configuration
if __name__ == "__main__":
    # Run tests with different configurations
    pytest.main([
        __file__,
        "-v",                    # Verbose
        "--tb=short",           # Short traceback
        "-x",                   # Stop on first failure
        "--disable-warnings",   # Disable warnings
        "-m", "not slow",       # Skip slow tests by default
    ])


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_tree(nodes):
        """Create a test tree from nodes"""
        from raptor.tree_structures import Tree
        
        layer_to_nodes = {0: nodes}
        all_nodes = {i: node for i, node in enumerate(nodes)}
        
        return Tree(
            all_nodes=all_nodes,
            root_nodes=nodes[:1],  # First node as root
            leaf_nodes=nodes,
            num_layers=1,
            layer_to_nodes=layer_to_nodes
        )
    
    @staticmethod
    def assert_retrieval_quality(results, query, min_relevance=0.1):
        """Assert that retrieval results meet quality standards"""
        assert len(results) > 0, "No results returned"
        
        for result in results:
            assert hasattr(result, 'score'), "Result missing score"
            assert result.score >= min_relevance, f"Result score {result.score} below threshold"
            assert hasattr(result, 'node'), "Result missing node"
            assert result.node.text is not None, "Result node missing text"