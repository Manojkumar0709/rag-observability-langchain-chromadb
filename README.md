# RAG Observability with LangChain, ChromaDB and Mistral 7B

A production-ready Retrieval-Augmented Generation (RAG) application with built-in observability. Ask questions over your own documents using a locally running Mistral 7B model via Ollama, with ChromaDB as the vector store and a Streamlit web interface.

---

## Features

- Load and ingest `.txt`, `.pdf`, and `.csv` documents automatically
- Chunk and embed documents using `sentence-transformers/all-MiniLM-L6-v2`
- Store and retrieve embeddings from a local ChromaDB vector store
- Answer questions using Mistral 7B running locally via Ollama
- Three retriever modes: similarity, MMR (diverse), and contextual compression
- Built-in observability: latency tracking, query logging, and metrics dashboard
- Clean Streamlit UI with a Chat tab and a Logs and Metrics tab
- All configuration via `.env` file — no hardcoded values

---

## Project Structure

```
rag-observability-langchain-chromadb/
|
|-- dataset/                    # Place your documents here
|   |-- sample_qa.csv
|   |-- ai_knowledge_base.txt
|   |-- ml_concepts.txt
|   |-- nlp_deep_dive.txt
|   |-- rag_advanced_topics.txt
|   |-- extended_qa_dataset.csv
|
|-- src/
|   |-- __init__.py             # Package marker
|   |-- ingest.py               # Document loading, chunking, embedding, ChromaDB storage
|   |-- retriever.py            # ChromaDB retriever with 3 modes
|   |-- rag_chain.py            # LangChain RetrievalQA chain with Mistral via Ollama
|   |-- observability.py        # Logging, metrics, and LangChain callback
|   |-- app.py                  # Streamlit frontend
|
|-- logs/                       # Auto-created: stores rag_logs.jsonl
|-- chroma_db/                  # Auto-created: stores ChromaDB embeddings
|-- requirement.txt
|-- .env.example
|-- README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Mistral 7B via Ollama (local) |
| RAG Framework | LangChain + LangChain Community |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Frontend | Streamlit |
| Document Parsing | PyPDF, CSVLoader, TextLoader |
| Configuration | python-dotenv |
| Logging | Custom JSONL logger + Pandas |

---

## Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.com) installed and running locally
- Mistral model pulled via Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull Mistral 7B model
ollama pull mistral

# Verify Ollama is running
ollama serve
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manojkumar0709/rag-observability-langchain-chromadb.git
cd rag-observability-langchain-chromadb
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
# Paths
DATA_DIR=dataset/
CHROMA_DB_DIR=chroma_db/
LOG_FILE=logs/rag_logs.jsonl

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Ollama LLM
OLLAMA_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434

# Retriever
RETRIEVER_MODE=similarity
TOP_K=5
SIMILARITY_THRESHOLD=0.3
```

---

## Usage

### Step 1: Add your documents

Place any `.txt`, `.pdf`, or `.csv` files inside the `dataset/` folder. Sample datasets are already included.

### Step 2: Ingest documents into ChromaDB

```bash
python -m src.ingest
```

This will load all documents, split them into chunks, embed them, and store them in the local `chroma_db/` folder.

### Step 3: Launch the Streamlit app

```bash
streamlit run src/app.py
```

Open your browser at `http://localhost:8501`

---

## How It Works

```
User Question
      |
   app.py              Streamlit UI captures the question
      |
   rag_chain.py        LangChain RetrievalQA chain is invoked
      |         \
 retriever.py   Mistral 7B    Retriever fetches top-K chunks from ChromaDB
      |         via Ollama    Mistral generates answer using retrieved context
   ingest.py
      |
  ChromaDB             Documents stored as vector embeddings on disk
      |
observability.py       Logs query, answer, sources, and latency to JSONL
```

---

## Retriever Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `similarity` | Standard cosine similarity search | Most queries |
| `mmr` | Maximal Marginal Relevance - returns diverse results | Broad topics |
| `compressed` | Filters out low-relevance chunks before passing to LLM | Precision answers |

Change the mode from the sidebar in the Streamlit app or set `RETRIEVER_MODE` in `.env`.

---

## Observability

Every query is automatically logged to `logs/rag_logs.jsonl` with the following fields:

```json
{
  "timestamp": "2026-07-14T18:00:00",
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": ["dataset/ai_knowledge_base.txt"],
  "num_sources": 1,
  "latency_seconds": 1.243,
  "metadata": {}
}
```

The Logs and Metrics tab in the app shows:
- Total queries run
- Average, min, and max latency
- Latency trend chart over time
- Full scrollable query history table

---

## Dataset

The `dataset/` folder includes the following sample files ready for ingestion:

| File | Description |
|------|-------------|
| `ai_knowledge_base.txt` | General AI and LLM concepts |
| `ml_concepts.txt` | Supervised, unsupervised, deep learning, optimization |
| `nlp_deep_dive.txt` | Tokenization, BERT, GPT, attention, fine-tuning |
| `rag_advanced_topics.txt` | Chunking, hybrid search, re-ranking, Graph RAG, evaluation |
| `sample_qa.csv` | 15 Q&A pairs on RAG, LangChain, ChromaDB |
| `extended_qa_dataset.csv` | 30 Q&A pairs with category and difficulty labels |

You can add your own documents — just drop them in the `dataset/` folder and re-run ingestion.

---

## Customization

### Switch to a different LLM

```env
# Use LLaMA 3 instead of Mistral
OLLAMA_MODEL=llama3
```

```bash
ollama pull llama3
```

### Adjust chunk size

```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

### Use more retrieved documents

```env
TOP_K=8
```

---

## File Reference

| File | Role |
|------|------|
| `src/ingest.py` | Load, chunk, embed, and store documents in ChromaDB |
| `src/retriever.py` | Load ChromaDB and return a configured retriever |
| `src/rag_chain.py` | Build RetrievalQA chain using Mistral via Ollama |
| `src/observability.py` | LangChain callback, query logger, and stats |
| `src/app.py` | Streamlit Chat and Metrics UI |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

ManojKumar MohanKumar

GitHub: [Manojkumar0709](https://github.com/Manojkumar0709)
