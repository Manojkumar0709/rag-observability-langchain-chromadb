"""Document ingestion pipeline: load, split, embed, and store in ChromaDB."""

import os
from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "dataset/")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def load_documents(data_dir: str) -> list:
    """Load all .txt, .pdf, and .csv files from the data directory."""
    docs = []

    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)
    docs.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
    docs.extend(pdf_loader.load())

    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            csv_path = os.path.join(data_dir, fname)
            csv_docs = CSVLoader(file_path=csv_path).load()
            docs.extend(csv_docs)

    print(f"[Ingest] Loaded {len(docs)} documents from '{data_dir}'")
    return docs


def split_documents(docs: list) -> list:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[Ingest] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()
    print(f"[Ingest] Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DB_DIR}'")
    return vectorstore


def run_ingestion():
    """Full ingestion pipeline."""
    docs = load_documents(DATA_DIR)
    if not docs:
        print("[Ingest] No documents found. Please add files to the dataset/ folder.")
        return None
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(chunks)
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
