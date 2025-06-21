import os
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# === Config ===
PDF_PATH = "./Vivek_B_Resume_DE2.pdf"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable")

# === Core Functions ===
def pdf_reader(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def textsplitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [doc.page_content for doc in splitter.create_documents([text])]

def init_vector_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class SBERTEmbeddingFunction:
        def __call__(self, input):
            if isinstance(input, str):
                input = [input]
            return model.encode(input).tolist()
        def name(self):
            return "sbert-mini"

    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=SBERTEmbeddingFunction())

def add_documents_to_collection(collection, chunks):
    if collection.count() == 0:
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)

def query_collection(collection, query, n_results=2):
    return collection.query(query_texts=[query], n_results=n_results)["documents"][0]

def generate_answer_with_together(prompt):
    response = requests.post("https://api.together.xyz/v1/completions", headers={
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.2,
        "stop": ["\n\n"]
    })
    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    return f"[Error] {response.status_code}: {response.text}"

# === Tools ===
def tool_summarize(text):
    print(f"Prompt sent to Together:\n{prompt}")
    prompt = f"Summarize the following resume:\n{text}\n\nSummary:"
    print(f"Prompt sent to Together:\n{prompt}")
    return generate_answer_with_together(prompt)


def tool_extract_skills(text):
    prompt = f"List the technical skills mentioned:\n{text}\n\nSkills:"
    return generate_answer_with_together(prompt)

def tool_extract_responsibilities(text):
    prompt = f"List the candidate's professional responsibilities:\n{text}\n\nResponsibilities:"
    return generate_answer_with_together(prompt)

def tool_compare_with_jd(resume_text, jd_text):
    prompt = f"Compare this resume with the job description below and rate the fit (0–100) with a short explanation.\nResume:\n{resume_text}\n\nJD:\n{jd_text}\n\nFit:"
    return generate_answer_with_together(prompt)

def tool_draft_reply(text):
    prompt = f"Write an email response inviting the candidate for an interview:\n{text}\n\nEmail:"
    return generate_answer_with_together(prompt)

def tool_generate_linkedin_message(text):
    prompt = f"Draft a LinkedIn connection message to a hiring manager:\n{text}\n\nMessage:"
    return generate_answer_with_together(prompt)

def tool_answer_question(context, query):
    prompt = f"""
Using the following context, answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:
"""
    return generate_answer_with_together(prompt)

# === Chain ===
def chain_resume_fit_analysis(resume_text):
    summary = tool_summarize(resume_text)
    skills = tool_extract_skills(resume_text)
    ideal_jd = "A data engineer should have skills in SQL, Spark, Python, AWS, data pipelines, and big data tools."
    fit = tool_compare_with_jd(resume_text, ideal_jd)
    return f"--- Summary ---\n{summary}\n\n--- Skills ---\n{skills}\n\n--- Fit ---\n{fit}"

# === Planner ===
def simple_planner(query):
    q = query.lower()
    if "summarize" in q:
        return "summarize"
    elif "skills" in q:
        return "extract_skills"
    elif "responsibilities" in q:
        return "extract_responsibilities"
    elif "job description" in q or "fit" in q:
        return "compare_with_jd"
    elif "email" in q or "reply" in q:
        return "draft_reply"
    elif "linkedin" in q:
        return "linkedin_message"
    elif "fit for" in q:
        return "chain_fit"
    else:
        return "general"

# === Main ===
def main():
    chat_history = []

    print("Reading PDF...")
    text = pdf_reader(PDF_PATH)
    if not text:
        print("No text found. Exiting.")
        return

    print("Splitting into chunks...")
    chunks = textsplitter(text)

    print("Initializing vector DB...")
    collection = init_vector_db()
    add_documents_to_collection(collection, chunks)

    while True:
        print(f"Total chunks: {len(chunks)}")
        query = input("\nEnter your question (or 'exit'): ")
        print(f"First chunk preview:\n{chunks[0][:300]}") 
        if query.lower() == "exit":
            break

        retrieved_chunks = query_collection(collection, query)
        context = "\n".join(retrieved_chunks)
        action = simple_planner(query)

        if action == "summarize":
            answer = tool_summarize(context)
        elif action == "extract_skills":
            answer = tool_extract_skills(context)
        elif action == "extract_responsibilities":
            answer = tool_extract_responsibilities(context)
        elif action == "compare_with_jd":
            print("Paste the job description:")
            jd_text = input()
            answer = tool_compare_with_jd(context, jd_text)
        elif action == "draft_reply":
            answer = tool_draft_reply(context)
        elif action == "linkedin_message":
            answer = tool_generate_linkedin_message(context)
        elif action == "chain_fit":
            answer = chain_resume_fit_analysis(context)
        else:
            answer = tool_answer_question(context, query)

        chat_history.append((query, answer))
        print("\n--- Agentic Answer ---")
        print(answer)

        print("\n--- Chat History ---")
        for i, (q, a) in enumerate(chat_history):
            print(f"Q{i+1}: {q}\nA{i+1}: {a}\n")

if __name__ == "__main__":
    main()
