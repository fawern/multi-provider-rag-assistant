# Hybrid LangChain RAG Assistant

A command-line RAG system supporting OpenAI and HuggingFace providers with automatic document ingestion.

## Setup

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Provider Combinations

### 1. OpenAI + OpenAI (Default - Best Quality)
```bash
python qa.py --question "How do I reset my password?"
python qa.py --question "What are the API endpoints?"
python qa.py --question "How do I configure webhooks?" 
python qa.py --question "What user roles are supported?"
```

### 2. OpenAI + HuggingFace (Fast Embeddings + Local LLM)
```bash
python qa.py --question "How do I reset my password?" --llm-provider huggingface --hf-model google/flan-t5-small
python qa.py --question "What are the API endpoints?" --llm-provider huggingface --hf-model distilgpt2
python qa.py --question "How do I configure webhooks?" --llm-provider huggingface --hf-model microsoft/DialoGPT-small
python qa.py --question "What user roles are supported?" --llm-provider huggingface --hf-model google/flan-t5-base
```

### 3. HuggingFace + OpenAI (Local Embeddings + Quality LLM)
```bash
python qa.py --question "How do I reset my password?" --embedding-provider huggingface
python qa.py --question "What are the API endpoints?" --embedding-provider huggingface --k 5
python qa.py --question "How do I configure webhooks?" --embedding-provider huggingface
python qa.py --question "What user roles are supported?" --embedding-provider huggingface
```

### 4. HuggingFace + HuggingFace (Fully Local - No API Key)
```bash
python qa.py --question "How do I reset my password?" --embedding-provider huggingface --llm-provider huggingface
python qa.py --question "What are the API endpoints?" --embedding-provider huggingface --llm-provider huggingface --hf-model google/flan-t5-small
python qa.py --question "How do I configure webhooks?" --embedding-provider huggingface --llm-provider huggingface --hf-model distilgpt2
python qa.py --question "What user roles are supported?" --embedding-provider huggingface --llm-provider huggingface --hf-model google/flan-t5-base
```

## Troubleshooting

If you get dimension errors, clear the vector store:
```bash
python clear_vectorstore.py
```