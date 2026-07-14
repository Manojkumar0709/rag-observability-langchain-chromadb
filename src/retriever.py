"""Retriever module: load ChromaDB and return a retriever with optional hybrid search."""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB vectorstore from disk."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    print(f"[Retriever] Loaded vectorstore from '{CHROMA_DB_DIR}'")
    return vectorstore


def get_base_retriever(vectorstore: Chroma):
    """Return a basic similarity retriever."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )


def get_mmr_retriever(vectorstore: Chroma):
    """Return a Maximal Marginal Relevance retriever for diverse results."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3}
    )


def get_compressed_retriever(vectorstore: Chroma):
    """Return a retriever with contextual compression to filter irrelevant chunks."""
    base_retriever = get_base_retriever(vectorstore)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=SIMILARITY_THRESHOLD)
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    print("[Retriever] Using contextual compression retriever")
    return compressed_retriever


def get_retriever(mode: str = "similarity"):
    """Factory function to get the appropriate retriever.
    
    Args:
        mode: 'similarity' | 'mmr' | 'compressed'
    """
    vectorstore = load_vectorstore()
    if mode == "mmr":
        return get_mmr_retriever(vectorstore)
    elif mode == "compressed":
        return get_compressed_retriever(vectorstore)
    else:
        return get_base_retriever(vectorstore)
