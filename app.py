"""
Kerala Community Development RAG Chatbot
=========================================
Gradio app for HuggingFace Spaces. Answers questions about 436 academic papers
on Kerala governance and development using FAISS + BGE embeddings + GPT.
Supports multi-turn conversation with query reformulation.
"""

import os
import gc
import re

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
6. Do NOT include a "Sources Used" or "References" section at the end — sources are displayed separately by the system.
7. Only cite sources that are directly relevant to the question. Ignore retrieved sources that are off-topic.
8. You have access to the conversation history. Use it to understand follow-up questions and maintain context across turns."""

REFORMULATE_PROMPT = """Given the conversation history below, rewrite the user's latest message as a \
standalone search query that captures the full context. The query will be used to search a database \
of academic papers on Kerala community development and governance.

Output ONLY the rewritten query, nothing else.

CONVERSATION HISTORY:
{history}

LATEST USER MESSAGE: {question}

STANDALONE QUERY:"""

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


def clean_title(raw_title):
    """Convert filenames like 'Some_Paper_Title_W3041277519.pdf' to readable titles."""
    t = str(raw_title or "Untitled")
    t = re.sub(r'_W\d+\.pdf$', '', t)
    t = re.sub(r'\.pdf$', '', t, flags=re.IGNORECASE)
    t = t.replace('_', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    return t if t else "Untitled"


def format_sources(docs):
    seen = {}
    for doc in docs:
        md = doc.metadata
        doc_id = md.get('doc_id', '?')
        if doc_id not in seen:
            seen[doc_id] = {
                "title": clean_title(md.get('display_title', '')),
                "year": md.get('year', '?'),
                "geo": md.get('geo_scope', '?'),
                "metric": md.get('metric_bucket_primary', '?'),
                "pages": [],
            }
        seen[doc_id]["pages"].append(str(md.get('page', '?')))

    lines = []
    for i, (doc_id, info) in enumerate(seen.items(), 1):
        pages = ", ".join(info["pages"])
        year = info["year"]
        if isinstance(year, float):
            year = int(year)
        lines.append(
            f"**{i}. {info['title']}** ({year})\n"
            f"   - Document: `{doc_id}` | Pages cited: {pages}\n"
            f"   - Region: {info['geo']} | Topic: {info['metric']}\n"
        )
    return "\n".join(lines)


def get_api_key(user_key):
    if user_key and user_key.strip():
        return user_key.strip()
    return os.environ.get("OPENAI_API_KEY", "")


def reformulate_query(history, question, api_key, model):
    """Rewrite a follow-up question as a standalone query using conversation context."""
    if not history:
        return question

    # Build a compact history string from the last 6 messages (3 turns)
    recent = history[-6:]
    history_str = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:300]}"
        for msg in recent
    )

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, max_tokens=150)
    prompt = ChatPromptTemplate.from_messages([
        ("human", REFORMULATE_PROMPT),
    ])
    chain = prompt | llm
    result = chain.invoke({"history": history_str, "question": question})
    standalone = result.content.strip()
    return standalone if standalone else question


def build_chat_messages(history, context, question):
    """Build the message list for the LLM, including conversation history."""
    messages = [("system", SYSTEM_PROMPT)]

    # Include the last 6 messages of conversation for context
    # Gradio uses "user"/"assistant", LangChain expects "human"/"ai"
    recent = history[-6:]
    for msg in recent:
        role = "human" if msg["role"] == "user" else "ai"
        messages.append((role, msg["content"]))

    # Final user message with retrieved sources
    messages.append((
        "human",
        "Based on the following source documents, answer the question.\n\n"
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a thorough, well-cited answer:"
    ))
    return messages


def render_conversation(history):
    """Render chat history as Markdown for display."""
    if not history:
        return "*No conversation yet. Ask a question to get started.*"
    parts = []
    for msg in history:
        if msg["role"] == "user":
            parts.append(f"**You:** {msg['content']}")
        else:
            parts.append(f"**Assistant:**\n\n{msg['content']}")
    return "\n\n---\n\n".join(parts)


def respond(message, chat_history, api_key, model, top_k):
    """Process a message with full conversation context."""
    resolved_key = get_api_key(api_key)
    if not resolved_key:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please enter your OpenAI API key in the sidebar."},
        ]
        return chat_history, render_conversation(chat_history), "", ""

    if not message or not message.strip():
        return chat_history, render_conversation(chat_history), "", ""

    # Step 1: Reformulate query using conversation history
    standalone_query = reformulate_query(chat_history, message, resolved_key, model)

    # Step 2: Retrieve from FAISS using the standalone query
    docs = vectorstore.similarity_search(standalone_query, k=int(top_k))
    if not docs:
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "No relevant documents found for your query."},
        ]
        return chat_history, render_conversation(chat_history), "", ""

    # Step 3: Build prompt with conversation history + retrieved context
    context = format_context(docs)
    messages = build_chat_messages(chat_history, context, message)
    prompt = ChatPromptTemplate.from_messages(messages)

    # Step 4: Generate answer
    llm = ChatOpenAI(model=model, temperature=0.1, api_key=resolved_key)
    chain = prompt | llm
    response = chain.invoke({})

    answer = response.content
    sources = format_sources(docs)

    # Step 5: Update chat history
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    return chat_history, render_conversation(chat_history), sources, ""


def clear_chat():
    """Reset conversation."""
    return [], "*No conversation yet. Ask a question to get started.*", "", ""


# ---------------------------------------------------------------------------
# Gradio UI — conversational interface
# ---------------------------------------------------------------------------
ENV_KEY_SET = bool(os.environ.get("OPENAI_API_KEY", ""))

with gr.Blocks(
    title="Kerala Community Development RAG",
    theme=gr.themes.Soft(primary_hue="green"),
    analytics_enabled=False,
) as demo:

    gr.Markdown(
        "# Kerala Community Development RAG\n"
        "Ask questions about **436 academic papers** on Kerala governance, "
        "decentralization, and development policy.\n\n"
        "Supports **follow-up questions** — ask a question, then dig deeper."
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
                "2. Follow-up questions are automatically reformulated for better search\n"
                "3. GPT generates an answer with `[doc_id, p.X]` citations\n"
            )
            gr.Markdown(
                "---\n**Try a conversation:**\n"
                "1. *What is Kudumbashree?*\n"
                "2. *How does it help women specifically?*\n"
                "3. *What about in Trivandrum?*\n"
            )

        with gr.Column(scale=3):
            # Hidden state for conversation history (list of dicts)
            chat_state = gr.State([])

            question = gr.Textbox(
                label="Your Question",
                placeholder="Ask about Kerala community development...",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary", size="lg")
                clear_btn = gr.Button("Clear Chat", variant="secondary", size="lg")

            gr.Markdown("### Conversation")
            conversation_box = gr.Markdown(
                value="*No conversation yet. Ask a question to get started.*"
            )
            gr.Markdown("### Sources (for latest answer)")
            sources_box = gr.Markdown()

            # Wire up events
            submit_btn.click(
                fn=respond,
                inputs=[question, chat_state, api_key, model, top_k],
                outputs=[chat_state, conversation_box, sources_box, question],
                api_name=False,
            )
            question.submit(
                fn=respond,
                inputs=[question, chat_state, api_key, model, top_k],
                outputs=[chat_state, conversation_box, sources_box, question],
                api_name=False,
            )
            clear_btn.click(
                fn=clear_chat,
                inputs=[],
                outputs=[chat_state, conversation_box, sources_box, question],
                api_name=False,
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
