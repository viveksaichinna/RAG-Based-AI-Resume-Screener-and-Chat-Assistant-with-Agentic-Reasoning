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

    class DummyEmbeddingFunction:
        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            return [[len(x)] for x in input]
        def name(self):
            return "dummy"

    return client.get_or_create_collection(name=collection_name, embedding_function=DummyEmbeddingFunction())

def add_documents(db_path, collection_name, chunks):
    import chromadb

    # Delete and recreate the collection to ensure it's clean
    client = chromadb.PersistentClient(path=db_path)

    # Delete collection if it exists
    existing = client.list_collections()
    if any(c.name == collection_name for c in existing):
        client.delete_collection(name=collection_name)

    # Recreate fresh collection
    class DummyEmbeddingFunction:
        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            return [[len(x)] for x in input]
        def name(self):
            return "dummy"

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=DummyEmbeddingFunction()
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return collection

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
