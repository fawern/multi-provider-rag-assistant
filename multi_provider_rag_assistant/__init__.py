"""
Hybrid LangChain RAG Assistant

A command-line RAG system supporting OpenAI, GROQ, and HuggingFace providers
with automatic document ingestion and fallback mechanisms.
"""

__version__ = "1.0.0"
__author__ = "Enes Ekinci"

from .config import ProviderConfig, ProviderFactory
from .qa import answer_question
from .ingest import DocumentIngestor

__all__ = ["ProviderConfig", "ProviderFactory", "answer_question", "DocumentIngestor"]