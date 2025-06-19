# streamlit_app.py

import streamlit as st
import os
import shutil
from main import *

DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"


st.set_page_config(page_title="RAG PDF Chat", layout="wide")

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None

st.sidebar.title("📚 RAG PDF Chat")
st.sidebar.markdown("Upload a PDF and start chatting!")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    pdf_path = os.path.join("pdfs", uploaded_file.name)
    os.makedirs("pdfs", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")

    # Process the PDF
    text = pdf_reader(pdf_path)
    chunks = textsplitter(text)

    # Recreate and load collection with fresh data
    collection = add_documents(DB_PATH, COLLECTION_NAME, chunks)
    st.session_state.collection = collection
    st.success("PDF processed and stored into fresh collection!")


# New chat button
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_history = []

st.title("🧠 Chat with your PDF")

# User input
user_query = st.text_input("Ask a question based on the PDF:")

if user_query and st.session_state.collection:
    # Query DB
    retrieved_chunks = query_collection(st.session_state.collection, user_query)
    context = "\n".join(retrieved_chunks)

    # Construct prompt
    prompt = f"""
Using the following context, answer the question. If the answer is not in the context,
say you don't know.

Context:
{context}

Question: {user_query}

Answer:
"""
    # Get response
    response = generate_answer(prompt)
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)
