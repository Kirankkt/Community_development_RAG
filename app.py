"""
Kerala Community Development RAG Chatbot
=========================================
Gradio app for HuggingFace Spaces. Answers questions about 436 academic papers
on Kerala governance and development using FAISS + BGE embeddings + GPT.
"""

import os
import pickle

import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index_bge_base")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunk_docs.pkl")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
# Load everything at startup (runs once when Space boots)
# ---------------------------------------------------------------------------
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading FAISS index...")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

print("Loading chunks for BM25...")
with open(CHUNKS_PATH, "rb") as f:
    chunk_docs = pickle.load(f)

print("Building BM25 index...")
bm25_retriever = BM25Retriever.from_documents(chunk_docs, k=25)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4],
)

print("Loading reranker...")
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

print("All models loaded!")


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
def retrieve_and_rerank(query: str, top_k: int = 6):
    candidates = ensemble_retriever.invoke(query)
    if not candidates:
        return []
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    results = []
    for doc, score in scored[:top_k]:
        doc.metadata["reranker_score"] = float(score)
        results.append(doc)
    return results


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


def format_sources(docs):
    lines = []
    for i, doc in enumerate(docs, 1):
        md = doc.metadata
        lines.append(
            f"**[{i}] {md.get('doc_id','?')}** p.{md.get('page','?')} "
            f"(score: {md.get('reranker_score', 0):.2f})\n"
            f"*{str(md.get('display_title',''))[:120]}* ({md.get('year','?')})\n"
            f"Geo: {md.get('geo_scope','?')} | Metric: {md.get('metric_bucket_primary','?')}\n"
            f"```\n{doc.page_content[:250]}...\n```\n"
        )
    return "\n".join(lines)


def rag_query(question: str, api_key: str, model: str, top_k: int):
    if not api_key or not api_key.strip().startswith("sk-"):
        return "Please enter a valid OpenAI API key.", ""

    if not question.strip():
        return "Please enter a question.", ""

    # Retrieve and rerank
    docs = retrieve_and_rerank(question, top_k=int(top_k))
    if not docs:
        return "No relevant documents found for your query.", ""

    # Generate
    context = format_context(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",
         "Based on the following source documents, answer the question.\n\n"
         "SOURCES:\n{context}\n\n"
         "QUESTION: {question}\n\n"
         "Provide a thorough, well-cited answer:"),
    ])
    llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key.strip())
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, format_sources(docs)


def chat_fn(message, history, api_key, model, top_k):
    """Gradio chat handler — yields streaming-style output."""
    answer, sources = rag_query(message, api_key, model, top_k)
    full = answer
    if sources:
        full += "\n\n---\n\n<details><summary>📚 View Retrieved Sources</summary>\n\n" + sources + "\n</details>"
    return full


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Kerala Community Development RAG",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.Markdown(
        "# 📚 Kerala Community Development RAG\n"
        "Ask questions about **436 academic papers** on Kerala governance, "
        "decentralization, and development policy.\n\n"
        "Uses **hybrid search** (FAISS + BM25), **cross-encoder reranking**, "
        "and **GPT** with source citations."
    )

    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                value=os.environ.get("OPENAI_API_KEY", ""),
            )
            model = gr.Dropdown(
                choices=["gpt-4o-mini", "gpt-4o"],
                value="gpt-4o-mini",
                label="LLM Model",
            )
            top_k = gr.Slider(
                minimum=3, maximum=10, value=6, step=1,
                label="Number of sources",
            )
            gr.Markdown(
                "---\n"
                "**How it works:**\n"
                "1. Your question is searched against 19,555 chunks from 436 papers\n"
                "2. Top candidates are re-ranked by a cross-encoder\n"
                "3. GPT generates an answer with `[doc_id, p.X]` citations\n"
            )

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[api_key, model, top_k],
                examples=[
                    "What is the role of Kudumbashree in poverty reduction?",
                    "What indicators measure municipal service delivery in Kerala?",
                    "How do gram panchayats contribute to decentralized governance?",
                    "What are the challenges in solid waste management in Kerala?",
                    "What is the impact of MGNREGA on rural employment in Kerala?",
                ],
                chatbot=gr.Chatbot(height=500, show_copy_button=True),
            )


if __name__ == "__main__":
    demo.launch()
