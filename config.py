import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    embedding_provider: str = "openai"
    llm_provider: str = "openai"
    huggingface_model: str = "google/flan-t5-small"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"
    groq_model: str = "openai/gpt-oss-120b"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "./vectorstore"
    docs_path: str = "./docs"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 3

class ProviderFactory:
    
    @staticmethod
    def get_embeddings(config: ProviderConfig):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        if config.embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=config.openai_embedding_model)
        
        elif config.embedding_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            try:
                return HuggingFaceEmbeddings(
                    model_name=config.hf_embedding_model,
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e:
                logger.error(f"Failed to load HuggingFace embeddings: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
    
    @staticmethod
    def get_llm(config: ProviderConfig):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        if config.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=config.openai_llm_model, temperature=0)
        
        elif config.llm_provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=config.groq_model, temperature=0)
        
        elif config.llm_provider == "huggingface":
            from langchain_huggingface import HuggingFacePipeline
            from transformers import pipeline
            
            try:
                if "t5" in config.huggingface_model.lower():
                    task = "text2text-generation"
                else:
                    task = "text-generation"
                
                pipe = pipeline(
                    task,
                    model=config.huggingface_model,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                logger.error(f"Failed to load HuggingFace LLM: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(levelname)s: %(message)s'
    )

def validate_config(config: ProviderConfig, check_llm: bool = True) -> bool:
    if config.embedding_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI API key required")
        return False
    
    if check_llm and config.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.error("OpenAI API key required")
        return False
    
    if check_llm and config.llm_provider == "groq" and not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ API key required")
        return False
    
    if not os.path.exists(config.docs_path):
        logger.error(f"Documents directory not found: {config.docs_path}")
        return False
    
    return True