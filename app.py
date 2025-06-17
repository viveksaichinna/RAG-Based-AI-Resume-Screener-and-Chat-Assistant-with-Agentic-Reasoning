import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Use HuggingFace local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Set Hugging Face API key (used for LLM, not embeddings in this case)
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Load LLM (e.g., FLAN-T5 or Mistral) from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # or another generation-supported model
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

st.title("📄 Chat with your PDF")

# Upload PDF
pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf:
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Create vector DB from chunks
    db = FAISS.from_texts(chunks, embeddings)

    # Input query
    query = st.text_input("Ask your PDF:")
    if query:
        docs = db.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=docs, question=query)
        st.write(result)
