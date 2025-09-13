import os
import logging
import click
import shutil
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from rich.console import Console

from config import ProviderConfig, ProviderFactory, setup_logging, validate_config

logger = logging.getLogger(__name__)
console = Console()

def get_or_create_vectorstore(config):
    
    if not os.path.exists(config.vector_store_path):
        logger.info("Creating vector store...")
        create_vectorstore(config)
    
    embeddings = ProviderFactory.get_embeddings(config)

    try:
        return FAISS.load_local(config.vector_store_path, embeddings, allow_dangerous_deserialization=True)
    except (AssertionError, Exception) as e:
        if "d == self.d" in str(e) or "dimension" in str(e).lower():
            logger.warning("Dimension mismatch. Rebuilding vector store...")
            if os.path.exists(config.vector_store_path):
                shutil.rmtree(config.vector_store_path)
            create_vectorstore(config)
            return FAISS.load_local(config.vector_store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            raise

def create_vectorstore(config):

    from ingest import DocumentIngestor

    ingestor = DocumentIngestor(config)
    ingestor.ingest()

def answer_question(question, config):

    vector_store = get_or_create_vectorstore(config)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})
    
    llm = ProviderFactory.get_llm(config)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    try:
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        error_str = str(e)
        if (
                "d == self.d" in error_str or 
                "dimension" in error_str.lower() or
                "assert d == self.d" in error_str or
                isinstance(e, AssertionError)
            ):
            
            logger.warning("Dimension mismatch detected. Rebuilding vector store...")
            if os.path.exists(config.vector_store_path):
                shutil.rmtree(config.vector_store_path)
            
            vector_store = get_or_create_vectorstore(config)
            retriever = vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain.invoke({"query": question})

            return result["result"]
        else:
            raise

@click.command()
@click.option('--question', required=True, help='Question to ask')
@click.option('--embedding-provider', default='openai', type=click.Choice(['openai', 'huggingface']), help='Embedding provider')
@click.option('--llm-provider', default='openai', type=click.Choice(['openai', 'groq', 'huggingface']), help='LLM provider')
@click.option('--hf-model', default='google/flan-t5-small', help='HuggingFace model')
@click.option('--k', default=3, type=int, help='Number of documents to retrieve')
@click.option('--log-level', default='WARNING', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Logging level')
def main(question, embedding_provider, llm_provider, hf_model, k, log_level):
        
    # Setup
    setup_logging(log_level)
    config = ProviderConfig(
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        huggingface_model=hf_model,
        retrieval_k=k
    )
    
    if not validate_config(config):
        print("Configuration error")
        return
    
    try:
        with console.status("Processing..."):
            answer = answer_question(question, config)
        
        print(f"Answer: {answer}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {str(e) if str(e) else 'Unknown error occurred'}")
        if log_level == 'DEBUG':
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()