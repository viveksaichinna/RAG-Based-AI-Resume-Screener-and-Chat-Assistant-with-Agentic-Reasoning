from langchain_huggingface import HuggingFaceEndpoint

# Community tools (embeddings, vector DBs, etc.)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import os

# # Set Hugging Face token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Load PDF
reader = PdfReader("/workspaces/explorellm/resume.pdf")


raw_text = "".join([page.extract_text() for page in reader.pages])

# Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)

# Embeddings and vector store
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts(texts, embeddings)

# Load LLM from HF Inference API
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

#prompt = PromptTemplate.from_template(
#    "Answer the question based on the following documents:\n\n{context}\n\nQuestion: {question}"
#)
prompt = "answer the given question"
chain = create_stuff_documents_chain(llm, prompt)

# Run query
query = "What is the main topic of the document?"
context = " resume of some person"
raw_docs = db.similarity_search(query)
from langchain.schema import Document

# assuming you have list of strings called raw_docs
docs = [Document(page_content=text) for text in raw_text]
response = chain.invoke({"input_documents": docs, "question": query, "context": context})

print(response)



