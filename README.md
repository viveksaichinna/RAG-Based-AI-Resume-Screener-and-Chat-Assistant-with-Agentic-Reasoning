# 🧠 RAG PDF Chat with Streamlit

This project allows users to upload PDF files and ask natural language questions about their content. It uses a **Retrieval-Augmented Generation (RAG)** approach with **ChromaDB** as the vector store, **Sentence Transformers** for generating embeddings, and **Together.ai** for generating responses using a powerful LLM.

## 🔍 Overview

- Upload a PDF and parse its content
- Split text into chunks for semantic search
- Generate real embeddings using `all-MiniLM-L6-v2`
- Store and query document chunks in ChromaDB
- Ask questions through a friendly Streamlit chat interface
- Generate context-aware answers using Together.ai LLM (e.g., Mistral-7B)

---

## 🧰 Tech Stack

| Component        | Technology                          |
|------------------|--------------------------------------|
| Frontend         | Streamlit                           |
| Backend          | Python, LangChain                   |
| Embeddings       | Sentence Transformers (`MiniLM`)    |
| Vector Store     | ChromaDB                            |
| LLM              | Together.ai (Mistral-7B Instruct)   |
| PDF Reader       | PyPDF                               |

---

