# Hybrid LangChain RAG Assistant

A command-line RAG system supporting OpenAI, GROQ, and HuggingFace providers with automatic document ingestion.

## Quick Install & Setup

```bash
# Clone the repository
git clone https://github.com/fawern/multi-provider-rag-assistant.git
cd multi-provider-rag-assistant

# Create virtual environment
python -m venv venv
# venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

# Optional: Add API keys (or use local HuggingFace fallback)
echo "OPENAI_API_KEY=your_key_here" > .env
echo "GROQ_API_KEY=your_groq_key_here" >> .env

# Test the system (works without API keys!)
python qa.py --question "How do I reset my password?"
```

## How It Works

The system uses two separate providers:

- **`--embedding-provider`**: Creates vector embeddings from documents and queries
  - `openai`: Uses OpenAI text-embedding-3-small (1536 dimensions)
  - `huggingface`: Uses sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)

- **`--llm-provider`**: Generates the final answer from retrieved context
  - `openai`: Uses OpenAI gpt-4o-mini (API-based)
  - `groq`: Uses GROQ openai/gpt-oss-120b (API-based)
  - `huggingface`: Uses local models like google/flan-t5-small (local inference)

## Setup

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
echo "GROQ_API_KEY=your_groq_key_here" >> .env
```

**Note**: API keys are optional! The system automatically falls back to HuggingFace (local) when API keys are missing.

## Provider Combinations

### 1. OpenAI Embeddings + OpenAI LLM (Default)
**Embeddings**: OpenAI text-embedding-3-small | **LLM**: OpenAI gpt-4o-mini
```bash
python qa.py --question "How do I reset my password?"
python qa.py --question "What are the API endpoints?"
python qa.py --question "How do I configure webhooks?" 
python qa.py --question "What user roles are supported?"
```

### 2. OpenAI Embeddings + GROQ LLM
**Embeddings**: OpenAI text-embedding-3-small | **LLM**: GROQ openai/gpt-oss-120b
```bash
python qa.py --question "How do I reset my password?" --llm-provider groq
python qa.py --question "What are the API endpoints?" --llm-provider groq
python qa.py --question "How do I configure webhooks?" --llm-provider groq
python qa.py --question "What user roles are supported?" --llm-provider groq
```

### 3. OpenAI Embeddings + HuggingFace LLM
**Embeddings**: OpenAI text-embedding-3-small | **LLM**: Local HuggingFace models
```bash
python qa.py --question "How do I reset my password?" --llm-provider huggingface --hf-model google/flan-t5-small
python qa.py --question "What are the API endpoints?" --llm-provider huggingface --hf-model distilgpt2
python qa.py --question "How do I configure webhooks?" --llm-provider huggingface --hf-model microsoft/DialoGPT-small
python qa.py --question "What user roles are supported?" --llm-provider huggingface --hf-model google/flan-t5-base
```

### 4. HuggingFace Embeddings + OpenAI LLM
**Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 | **LLM**: OpenAI gpt-4o-mini
```bash
python qa.py --question "How do I reset my password?" --embedding-provider huggingface
python qa.py --question "What are the API endpoints?" --embedding-provider huggingface --k 5
python qa.py --question "How do I configure webhooks?" --embedding-provider huggingface
python qa.py --question "What user roles are supported?" --embedding-provider huggingface
```

### 5. HuggingFace Embeddings + HuggingFace LLM (Fully Local)
**Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2 | **LLM**: Local HuggingFace models
```bash
python qa.py --question "How do I reset my password?" --embedding-provider huggingface --llm-provider huggingface
python qa.py --question "What are the API endpoints?" --embedding-provider huggingface --llm-provider huggingface --hf-model google/flan-t5-small
python qa.py --question "How do I configure webhooks?" --embedding-provider huggingface --llm-provider huggingface --hf-model distilgpt2
python qa.py --question "What user roles are supported?" --embedding-provider huggingface --llm-provider huggingface --hf-model google/flan-t5-base
```

## Automatic Fallback System

The system automatically falls back to HuggingFace when API keys are missing:

- **Missing OpenAI key**: Falls back to `huggingface` for embeddings/LLM
- **Missing GROQ key**: Falls back to `huggingface` for LLM  
- **No API keys needed**: Use `--embedding-provider huggingface --llm-provider huggingface` for fully local operation

**Example without any API keys:**
```bash
# This works even without .env file
python qa.py --question "How do I reset my password?"
# WARNING: OpenAI API key not found for embeddings. Falling back to HuggingFace embeddings.
# WARNING: OpenAI API key not found for LLM. Falling back to HuggingFace LLM.
# Using providers: embeddings=huggingface, llm=huggingface
```

## Troubleshooting

If you get dimension errors, clear the vector store:
```bash
python clear_vectorstore.py
```