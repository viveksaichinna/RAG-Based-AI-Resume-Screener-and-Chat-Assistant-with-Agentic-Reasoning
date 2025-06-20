import os
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

PDF_FOLDER = "./pdfs"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def pdf_reader(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def textsplitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return [doc.page_content for doc in splitter.create_documents([text])]

def init_vector_db(db_path=DB_PATH, collection_name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=db_path)

    # Load real embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class SBERTEmbeddingFunction:
        def _call_(self, input: list[str]) -> list[list[float]]:
            if isinstance(input, str):  # Optional, defensive
                input = [input]
            return model.encode(input).tolist()

        def name(self):
            return "sbert-mini"

    embedding_function = SBERTEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return collection
# 4. Add documents (chunks) to ChromaDB
def add_documents_to_collection(collection, chunks):
    if collection.count() == 0:
        ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
        print(f"Adding {len(chunks)} chunks to the vector DB...")
        collection.add(documents=chunks, ids=ids)
    else:
        print("Collection already populated.")

def query_collection(collection, query, n_results=2):
    return collection.query(query_texts=[query], n_results=n_results)["documents"][0]

def generate_answer(prompt):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "stop": ["\n\n"]
    }
    response = requests.post("https://api.together.xyz/v1/completions", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    else:
        print("Error:", response.status_code, response.text)
        return "Error in LLM response."
