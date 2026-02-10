"""
Kerala Community Development RAG Chatbot
=========================================
Gradio app for HuggingFace Spaces. Answers questions about 436 academic papers
on Kerala governance and development using FAISS + BGE embeddings + GPT.
"""

import os
import gc

import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index_bge_base")
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
# Load at startup
# ---------------------------------------------------------------------------
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Loading FAISS index...")
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

gc.collect()
print("All models loaded!")


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------
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
            f"**[{i}] {md.get('doc_id','?')}** p.{md.get('page','?')}\n"
            f"*{str(md.get('display_title',''))[:120]}* ({md.get('year','?')})\n"
            f"Geo: {md.get('geo_scope','?')} | Metric: {md.get('metric_bucket_primary','?')}\n"
            f"```\n{doc.page_content[:250]}...\n```\n"
        )
    return "\n".join(lines)


def get_api_key(user_key: str) -> str:
    if user_key and user_key.strip():
        return user_key.strip()
    return os.environ.get("OPENAI_API_KEY", "")


def respond(message, history, api_key, model, top_k):
    resolved_key = get_api_key(api_key)
    if not resolved_key:
        history.append((message, "Please enter your OpenAI API key in the sidebar."))
        return history, ""

    if not message.strip():
        return history, ""

    # Retrieve from FAISS
    docs = vectorstore.similarity_search(message, k=int(top_k))
    if not docs:
        history.append((message, "No relevant documents found for your query."))
        return history, ""

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
    llm = ChatOpenAI(model=model, temperature=0.1, api_key=resolved_key)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": message})

    answer = response.content
    sources = format_sources(docs)
    full_response = answer + "\n\n---\n**Retrieved Sources:**\n\n" + sources

    history.append((message, full_response))
    return history, ""


# ---------------------------------------------------------------------------
# Gradio UI — manual chat layout (avoids ChatInterface bugs)
# ---------------------------------------------------------------------------
ENV_KEY_SET = bool(os.environ.get("OPENAI_API_KEY", ""))

with gr.Blocks(
    title="Kerala Community Development RAG",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.Markdown(
        "# Kerala Community Development RAG\n"
        "Ask questions about **436 academic papers** on Kerala governance, "
        "decentralization, and development policy.\n\n"
        "Uses **FAISS semantic search** (BGE embeddings) "
        "and **GPT** with source citations."
    )

    with gr.Row():
        with gr.Column(scale=1):
            if ENV_KEY_SET:
                gr.Markdown("**OpenAI API Key:** Set via environment secret")
            api_key = gr.Textbox(
                label="OpenAI API Key (leave blank if set via secret)" if ENV_KEY_SET else "OpenAI API Key",
                type="password",
                placeholder="Using environment secret" if ENV_KEY_SET else "sk-...",
                value="",
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
                "2. Best matching passages are retrieved via semantic search\n"
                "3. GPT generates an answer with `[doc_id, p.X]` citations\n"
            )
            gr.Markdown(
                "---\n**Example questions:**\n"
                "- What is the role of Kudumbashree in poverty reduction?\n"
                "- What indicators measure municipal service delivery in Kerala?\n"
                "- How do gram panchayats contribute to decentralized governance?\n"
                "- What are the challenges in solid waste management in Kerala?\n"
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=550, show_copy_button=True, label="Chat")
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask about Kerala community development...",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

            submit_btn.click(
                respond,
                inputs=[msg, chatbot, api_key, model, top_k],
                outputs=[chatbot, msg],
            )
            msg.submit(
                respond,
                inputs=[msg, chatbot, api_key, model, top_k],
                outputs=[chatbot, msg],
            )
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
