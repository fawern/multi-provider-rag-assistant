import os
import logging
import click
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

from config import ProviderConfig, ProviderFactory, setup_logging, validate_config, apply_fallback_providers

logger = logging.getLogger(__name__)

class DocumentIngestor:
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def load_documents(self):

        loader = DirectoryLoader(
            self.config.docs_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        return documents
    
    def create_vector_store(self, documents):
        
        chunks = self.text_splitter.split_documents(documents)
        
        embeddings = ProviderFactory.get_embeddings(self.config)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        os.makedirs(self.config.vector_store_path, exist_ok=True)
        vector_store.save_local(self.config.vector_store_path)
        
        return vector_store
    
    def ingest(self):
        documents = self.load_documents()
        if not documents:
            raise ValueError("No documents found")
        
        return self.create_vector_store(documents)

@click.command()
@click.option('--embedding-provider', default='openai', type=click.Choice(['openai', 'huggingface']), help='Embedding provider')
@click.option('--log-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Logging level')
def main(embedding_provider, log_level):
    setup_logging(log_level)
    
    config = ProviderConfig(embedding_provider=embedding_provider)
    
    config = apply_fallback_providers(config, check_llm=False)
    
    if not validate_config(config, check_llm=False):
        print("Configuration error")
        return
    
    try:
        ingestor = DocumentIngestor(config)
        ingestor.ingest()
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()