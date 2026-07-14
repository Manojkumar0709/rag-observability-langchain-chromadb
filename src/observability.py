"""Observability module: track, log, and monitor RAG pipeline metrics."""

import time
import json
import os
from datetime import datetime
from typing import Any
import pandas as pd
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = os.getenv("LOG_FILE", "logs/rag_logs.jsonl")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


class RAGObservabilityCallback(BaseCallbackHandler):
    """Custom LangChain callback to track LLM and chain metrics."""

    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "latency_seconds": None,
            "num_tokens_prompt": 0,
            "num_tokens_completion": 0,
            "llm_calls": 0,
            "errors": []
        }

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        self.metrics["start_time"] = time.time()
        self.metrics["llm_calls"] += 1
        print(f"[Observability] LLM call #{self.metrics['llm_calls']} started")

    def on_llm_end(self, response: Any, **kwargs):
        self.metrics["end_time"] = time.time()
        if self.metrics["start_time"]:
            self.metrics["latency_seconds"] = round(
                self.metrics["end_time"] - self.metrics["start_time"], 3
            )
        print(f"[Observability] LLM call completed in {self.metrics['latency_seconds']}s")

    def on_llm_error(self, error: Exception, **kwargs):
        self.metrics["errors"].append(str(error))
        print(f"[Observability] LLM error: {error}")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        print(f"[Observability] Chain started with inputs: {list(inputs.keys())}")

    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"[Observability] Chain completed with outputs: {list(outputs.keys())}")


def log_query(question: str, answer: str, sources: list, latency: float, metadata: dict = None):
    """Log a RAG query and its result to a JSONL file."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "sources": [doc.metadata.get("source", "Unknown") for doc in sources],
        "num_sources": len(sources),
        "latency_seconds": latency,
        "metadata": metadata or {}
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"[Observability] Logged query to '{LOG_FILE}'")
    return log_entry


def load_logs() -> pd.DataFrame:
    """Load all logged queries as a DataFrame."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    records = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def get_summary_stats() -> dict:
    """Return summary statistics of logged queries."""
    df = load_logs()
    if df.empty:
        return {"total_queries": 0}
    return {
        "total_queries": len(df),
        "avg_latency_seconds": round(df["latency_seconds"].mean(), 3),
        "max_latency_seconds": round(df["latency_seconds"].max(), 3),
        "min_latency_seconds": round(df["latency_seconds"].min(), 3),
        "avg_num_sources": round(df["num_sources"].mean(), 2),
    }
