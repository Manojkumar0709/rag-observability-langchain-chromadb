"""Streamlit frontend for the RAG Observability application."""

import time
import streamlit as st
from src.ingest import run_ingestion
from src.rag_chain import get_rag_chain, query_rag
from src.observability import log_query, load_logs, get_summary_stats

st.set_page_config(
    page_title="RAG Observability",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Configuration")
    retriever_mode = st.selectbox(
        "Retriever Mode",
        options=["similarity", "mmr", "compressed"],
        index=0,
        help="similarity: standard, mmr: diverse results, compressed: filtered"
    )
    top_k = st.slider("Top-K Documents", min_value=1, max_value=10, value=5)
    st.divider()
    st.markdown("### Data Ingestion")
    if st.button("Re-ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents into ChromaDB..."):
            run_ingestion()
        st.success("Documents ingested successfully!")
    st.divider()
    st.markdown("### Observability Stats")
    stats = get_summary_stats()
    st.metric("Total Queries", stats.get("total_queries", 0))
    st.metric("Avg Latency (s)", stats.get("avg_latency_seconds", "-"))
    st.metric("Avg Sources Used", stats.get("avg_num_sources", "-"))

# Main area
st.title("RAG Observability Dashboard")
st.caption("Powered by LangChain - ChromaDB - Ollama - Sentence Transformers")

tabs = st.tabs(["Chat", "Logs and Metrics"])

# Chat Tab
with tabs[0]:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        with st.spinner("Loading RAG chain..."):
            st.session_state.rag_chain = get_rag_chain()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src}`")

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                response = query_rag(prompt, chain=st.session_state.rag_chain)
                latency = round(time.time() - start, 3)

            answer = response["result"]
            sources = response.get("source_documents", [])

            st.markdown(answer)
            st.caption(f"Response time: {latency}s | Sources used: {len(sources)}")

            with st.expander("Sources"):
                for doc in sources:
                    src = doc.metadata.get("source", "Unknown")
                    st.markdown(f"- `{src}`")
                    st.caption(doc.page_content[:200] + "...")

            log_query(prompt, answer, sources, latency)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [doc.metadata.get("source", "Unknown") for doc in sources]
            })

# Logs and Metrics Tab
with tabs[1]:
    st.subheader("Query Logs")
    df = load_logs()
    if df.empty:
        st.info("No queries logged yet. Start chatting to generate logs!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", len(df))
        col2.metric("Avg Latency", f"{df['latency_seconds'].mean():.2f}s")
        col3.metric("Max Latency", f"{df['latency_seconds'].max():.2f}s")
        col4.metric("Avg Sources", f"{df['num_sources'].mean():.1f}")

        st.divider()
        st.subheader("Latency Over Time")
        st.line_chart(df.set_index("timestamp")["latency_seconds"])

        st.divider()
        st.subheader("Query History")
        st.dataframe(
            df[["timestamp", "question", "answer", "num_sources", "latency_seconds"]],
            use_container_width=True
        )
