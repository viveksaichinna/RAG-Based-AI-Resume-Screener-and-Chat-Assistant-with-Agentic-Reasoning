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

## 📁 Project Structure

explorellm/
│
├── pdfs/ # Folder for uploaded PDFs
├── chroma_db/ # Persistent vector DB
├── main.py # PDF processing and embedding code
├── streamlit_app.py # Streamlit web UI
├── requirements.txt # Python dependencies
└── README.md # This file

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/explorellm.git
cd explorellm
```

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```bash
pip install -r requirements.txt


```bash
export TOGETHER_API_KEY=your_together_api_key  # For Linux/macOS
# OR on Windows CMD
set TOGETHER_API_KEY=your_together_api_key

```bash
streamlit run app.py



##🧪 Example Usage
Upload any PDF (resume, research paper, report, etc.)

Ask questions like:

"What are the skills listed?"

"When was the certificate issued?"

"What is the paper title?"

The app retrieves relevant content from the PDF and uses the LLM to answer your question in context.


##📝 Features
✅ Multi-page PDF support

✅ Real embeddings (SBERT)

✅ ChromaDB vector search

✅ Together.ai LLM integration

✅ Streamlit UI with chat history

✅ Automatic collection reset on new PDF upload


##❗ Important Notes
Embeddings: We use sentence-transformers/all-MiniLM-L6-v2 for semantic similarity.

LLM Limitations: Together.ai has rate limits. You can switch to OpenAI or another provider with slight code changes.

Security: Don’t expose your API keys in shared or public environments.

🧠 What is RAG?
Retrieval-Augmented Generation (RAG) combines:

Retrieval – Finding relevant chunks of data (from PDFs here).

Generation – Using a language model to form an answer based on the retrieved content.

🔒 License
This project is released under the MIT License.

🙌 Credits
LangChain

Sentence Transformers

ChromaDB

Together.ai

Streamlit

🤝 Contributing
Pull requests are welcome! If you’d like to improve the UI, add support for multiple PDFs, or try another LLM, feel free to fork and enhance.

📬 Contact
Author: Vivek Sai Chinna Burada
Email: viveksaichinnaburada@gmail.com
LinkedIn: linkedin.com/in/viveksaichinna

vbnet
Copy
Edit

Let me know if you'd like to adjust this for **Docker support**, **multiple PDFs**, or **deployment** (Streamlit Cloud, HuggingFace Spaces, etc.).













