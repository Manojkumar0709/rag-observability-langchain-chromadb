"""RAG chain: combines retriever + LLM (Ollama) to answer questions."""

import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "similarity")

RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following retrieved context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this question."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""


def get_llm(streaming: bool = False):
    """Initialize the Ollama LLM."""
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        callbacks=callbacks,
        temperature=0.1,
    )
    print(f"[RAG Chain] Using LLM: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
    return llm


def get_rag_chain(streaming: bool = False):
    """Build and return the RAG chain."""
    retriever = get_retriever(mode=RETRIEVER_MODE)
    llm = get_llm(streaming=streaming)

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("[RAG Chain] RAG chain initialized successfully")
    return chain


def query_rag(question: str, chain=None) -> dict:
    """Run a query through the RAG chain.

    Returns:
        dict with 'result' (answer) and 'source_documents'
    """
    if chain is None:
        chain = get_rag_chain()
    response = chain.invoke({"query": question})
    return response


if __name__ == "__main__":
    question = "What is RAG and how does it work?"
    response = query_rag(question)
    print("\nAnswer:", response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        print(f"  - {doc.metadata.get('source', 'Unknown')}")
