import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "uploads"

documents = []

# Load all PDFs automatically
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)  
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

documents = text_splitter.split_documents(documents)

# Create embeddings and vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(documents)

db = FAISS.from_documents(documents, embeddings)
db.save_local("college_index")

print("Vector database created successfully!")
