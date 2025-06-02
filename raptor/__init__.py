# raptor/__init__.py
from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import (BaseEmbeddingModel,CustomEmbeddingModel)
from .FaissRetriever import FaissRetriever, FaissRetrieverConfig
from .QAModels import (BaseQAModel,GPT41QAModel, GPT4OMINIQAModel, GPT4QAModel)
from .RetrievalAugmentation import (RetrievalAugmentation,
                                    RetrievalAugmentationConfig)
from .Retrievers import BaseRetriever
from .SummarizationModels import (BaseSummarizationModel,
                                  GPT4OMiniSummarizationModel,
                                  GPT4OSummarizationModel,
                                  GPT41MiniSummarizationModel,
                                  GPT41SummarizationModel,
                                  )
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree


# Enhanced RAPTOR Hybrid Features - ADD TO END OF FILE
from .enhanced_retrieval_augmentation import (
    EnhancedRetrievalAugmentation, 
    HybridConfig, 
    create_enhanced_raptor
)
from .hybrid_retriever import (
    HybridRetriever, 
    FusionMethod, 
    HybridRetrievalResult,
    create_hybrid_retriever
)
from .sparse_retriever import (
    AdvancedBM25Retriever, 
    SparseRetrievalResult,
    create_sparse_retriever
)
from .query_enhancement import (
    QueryEnhancer, 
    EnhancedQuery,
    QueryIntent,
    create_query_enhancer
)
# ✅ Optional import:
try:
    from .evaluation_framework import (
        HybridRAPTOREvaluator,
        EvaluationQuery,
        create_sample_evaluation_set
    )
except ImportError:
    # Evaluation framework requires pandas, matplotlib, seaborn
    HybridRAPTOREvaluator = None
    EvaluationQuery = None
    create_sample_evaluation_set = None