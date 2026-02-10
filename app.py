"""
Kerala Community Development RAG Chatbot
=========================================
Streamlit app that answers questions about Kerala community development
using 436 academic papers with hybrid search, reranking, and GPT citations.
"""

import os
import pickle
import gc

import numpy as np
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index_bge_base")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunk_docs.pkl")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

SYSTEM_PROMPT = """You are an expert research assistant specializing in Kerala community development, \
local governance, decentralization, and Indian urban/rural development policy.

You answer questions using ONLY the provided source documents. Follow these rules strictly:

1. Base every claim on the provided sources. Cite using [doc_id, p.X] format after each claim.
2. If the sources do not contain enough information to answer, say so explicitly.
3. Synthesize information across multiple sources when relevant.
4. When sources disagree, note the disagreement and cite both sides.
5. Keep answers focused, well-structured, and academic in tone.
6. End with a "Sources Used" section listing all cited documents with their titles."""


# ---------------------------------------------------------------------------
# Load resources (cached so they only load once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_vectorstore(_embeddings):
    return FAISS.load_local(FAISS_DIR, _embeddings, allow_dangerous_deserialization=True)


@st.cache_resource(show_spinner="Loading document chunks...")
def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return chunks


# ---------------------------------------------------------------------------
# RAG pipeline functions
# ---------------------------------------------------------------------------
def retrieve(query, vectorstore, top_k=10):
    """Dense retrieval from FAISS, return top_k."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    docs = []
    for doc, score in results:
        doc.metadata["similarity_score"] = float(score)
        docs.append(doc)
    return docs


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        md = doc.metadata
        header = (
            f"[Source {i}] doc_id={md.get('doc_id','?')} | "
            f"page={md.get('page','?')} | "
            f"title={str(md.get('display_title',''))[:100]} | "
            f"year={md.get('year','?')} | "
            f"geo={md.get('geo_scope','?')}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def rag_query(question, vectorstore, llm, top_k=6):
    """Full RAG pipeline: retrieve + generate."""
    docs = retrieve(question, vectorstore, top_k=top_k)

    if not docs:
        return {
            "answer": "No relevant documents found for your query.",
            "sources": [],
            "docs": [],
        }

    context = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Based on the following source documents, answer the question.\n\n"
                "SOURCES:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "Provide a thorough, well-cited answer:",
            ),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    sources = []
    for doc in docs:
        md = doc.metadata
        sources.append(
            {
                "doc_id": md.get("doc_id", "?"),
                "page": md.get("page", "?"),
                "title": str(md.get("display_title", ""))[:120],
                "year": md.get("year", "?"),
                "geo_scope": md.get("geo_scope", "?"),
                "metric": md.get("metric_bucket_primary", "?"),
                "score": md.get("similarity_score", 0),
                "snippet": doc.page_content[:300],
            }
        )

    return {"answer": response.content, "sources": sources, "docs": docs}


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Kerala Community Development RAG",
        page_icon="📚",
        layout="wide",
    )

    st.title("Kerala Community Development RAG")
    st.caption("Ask questions about 436 academic papers on Kerala governance, development, and policy")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key. On Streamlit Cloud, set this in Secrets.",
        )
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                api_key = ""

        model_choice = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o"],
            help="gpt-4o-mini is faster and cheaper. gpt-4o gives higher quality answers.",
        )

        top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=6)

        st.divider()
        st.header("About")
        st.markdown(
            "This chatbot uses **dense semantic search** (BGE embeddings + FAISS) "
            "and **GPT** to answer questions "
            "grounded in academic literature on Kerala community development.\n\n"
            "Every claim includes a citation `[doc_id, p.X]` "
            "so you can trace it back to the source paper."
        )

    # --- Check prerequisites ---
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to get started.")
        st.stop()

    if not os.path.isdir(FAISS_DIR) or not os.path.isfile(CHUNKS_PATH):
        st.error(
            f"Data files not found in `{DATA_DIR}/`. "
            "Please run the export cell in the Colab notebook and unzip "
            "`rag_app_data.zip` into the `data/` folder."
        )
        st.stop()

    # --- Load models and data ---
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)

    llm = ChatOpenAI(model=model_choice, temperature=0.1, api_key=api_key)

    # --- Chat interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"View {len(msg['sources'])} sources"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**[{i}] {s['doc_id']}** p.{s['page']} "
                            f"(score: {s['score']:.4f})  \n"
                            f"*{s['title']}* ({s['year']})  \n"
                            f"Geo: {s['geo_scope']} | Metric: {s['metric']}  \n"
                            f"```\n{s['snippet']}...\n```"
                        )

    # User input
    if question := st.chat_input("Ask about Kerala community development..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching papers and generating answer..."):
                result = rag_query(question, vectorstore, llm, top_k=top_k)

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander(f"View {len(result['sources'])} sources"):
                    for i, s in enumerate(result["sources"], 1):
                        st.markdown(
                            f"**[{i}] {s['doc_id']}** p.{s['page']} "
                            f"(score: {s['score']:.4f})  \n"
                            f"*{s['title']}* ({s['year']})  \n"
                            f"Geo: {s['geo_scope']} | Metric: {s['metric']}  \n"
                            f"```\n{s['snippet']}...\n```"
                        )

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )


if __name__ == "__main__":
    main()
