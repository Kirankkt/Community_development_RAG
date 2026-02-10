---
title: Kerala Community Development RAG
emoji: 📚
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Kerala Community Development RAG

Ask questions about **436 academic papers** on Kerala governance, decentralization, and development policy.

**Pipeline:** Hybrid search (FAISS + BM25) → Cross-encoder reranking → GPT answer with `[doc_id, p.X]` citations.
