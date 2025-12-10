"""
Dependency injection for API endpoints.
Initialize models, RAG engine, and LLM on startup.
"""

import logging
from functools import lru_cache
# Import heavy model classes lazily to avoid blocking startup
_HEAVY_IMPORTS_PERFORMED = False

def _lazy_imports():
    global _HEAVY_IMPORTS_PERFORMED, InsuranceEnsembleModel, RAGEngine, OllamaFineTuner, RAGGenerator
    if _HEAVY_IMPORTS_PERFORMED:
        return
    from ml.models.ensemble import InsuranceEnsembleModel
    from ml.rag.retrieval import RAGEngine
    from ml.models.llm_fine_tune import OllamaFineTuner
    from ml.rag.generation import RAGGenerator
    _HEAVY_IMPORTS_PERFORMED = True

logger = logging.getLogger(__name__)

# Global instances
_ensemble_model = None
_rag_engine = None
_llm_engine = None
_rag_generator = None

async def init_models():
    """Stub initialization for models on startup.

    Do not instantiate heavy model classes here. We perform lazy
    initialization on first use to avoid long blocking startup times.
    """
    global _ensemble_model
    logger.info("Skipping heavy model instantiation at startup (initialized on first request)")
    _ensemble_model = None

async def init_rag():
    """Stub initialization for RAG engine on startup.

    Real RAG initialization is performed on first use to avoid heavy startup.
    """
    global _rag_engine
    logger.info("Skipping heavy RAG instantiation at startup (initialized on first request)")
    _rag_engine = None

def get_ensemble_model() -> 'InsuranceEnsembleModel':
    """Get ensemble model instance, performing lazy import/initialization."""
    global _ensemble_model
    if _ensemble_model is None:
        _lazy_imports()
        # instantiate only when first requested
        try:
            _ensemble_model = InsuranceEnsembleModel()
            try:
                _ensemble_model.load()
            except FileNotFoundError:
                logger.warning("No saved ensemble model found; will train on demand")
        except Exception as e:
            logger.error(f"Failed to instantiate ensemble model: {e}")
            raise
    return _ensemble_model

def get_rag_engine() -> 'RAGEngine':
    """Get RAG engine instance, performing lazy import/initialization."""
    global _rag_engine
    if _rag_engine is None:
        _lazy_imports()
        try:
            _rag_engine = RAGEngine()
        except Exception as e:
            logger.error(f"Failed to instantiate RAG engine: {e}")
            raise
    return _rag_engine

def get_llm_engine() -> 'OllamaFineTuner':
    """Get LLM fine-tuner instance, performing lazy import/initialization."""
    global _llm_engine
    if _llm_engine is None:
        _lazy_imports()
        try:
            _llm_engine = OllamaFineTuner()
        except Exception as e:
            logger.error(f"Failed to instantiate LLM engine: {e}")
            raise
    return _llm_engine

def get_rag_generator() -> 'RAGGenerator':
    """Get RAG generator instance, performing lazy import/initialization."""
    global _rag_generator
    if _rag_generator is None:
        _lazy_imports()
        try:
            _rag_generator = RAGGenerator()
        except Exception as e:
            logger.error(f"Failed to instantiate RAG generator: {e}")
            raise
    return _rag_generator
