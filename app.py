from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import shutil
import os
import uvicorn

# =============================
# APP INIT
# =============================
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

templates = Jinja2Templates(directory="templates")

# =============================
# CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# EMBEDDINGS (LAZY LOAD)
# =============================
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

# =============================
# LOAD VECTOR DATABASE
# =============================
def load_db():
    if os.path.exists("college_index"):
        print("Loading existing vector DB...")
        return FAISS.load_local(
            "college_index",
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    print("Vector DB not found.")
    return None

db = load_db()

# =============================
# BM25 GLOBALS
# =============================
bm25 = None
documents_list = []

def build_bm25():
    global bm25, documents_list
    if db is None:
        return
    documents_list = list(db.docstore._dict.values())
    corpus = [doc.page_content.split() for doc in documents_list]
    bm25 = BM25Okapi(corpus)
    print("BM25 index built successfully.")

if db:
    build_bm25()

# =============================
# REQUEST MODEL
# =============================
class Question(BaseModel):
    text: str

# =============================
# QUERY NORMALIZATION
# =============================
def normalize_query(query: str):
    query = query.lower().strip()

    replacements = {
        "operating sys": "operating systems",
        "os": "operating systems",
        "dbms": "database management systems",
        "cn": "computer networks",
        "ds": "data structures",
        "se": "software engineering",
        "ai": "artificial intelligence",
        "ml": "machine learning"
    }

    for short, full in replacements.items():
        if short in query:
            query = query.replace(short, full)

    return query

# =============================
# COURSE DETECTION
# =============================
def detect_course(user_query):
    course_keywords = [
        "operating systems",
        "database management systems",
        "computer networks",
        "data structures",
        "software engineering",
        "artificial intelligence",
        "machine learning"
    ]

    for course in course_keywords:
        if course in user_query:
            return course
    return None

# =============================
# ADMIN PANEL
# =============================
from pypdf import PdfReader

@app.get("/admin", response_class=HTMLResponse)
def admin_panel(request: Request, msg: str = None):

    files = os.listdir("uploads") if os.path.exists("uploads") else []
    total_pages = 0

    for file in files:
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join("uploads", file))
                total_pages += len(reader.pages)
            except Exception as e:
                print("Error reading:", file, e)

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "files": files,
            "message": msg,
            "pdf_count": len(files),
            "total_pages": total_pages
        }
    )

# =============================
# UPLOAD PDF
# =============================
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return RedirectResponse(
        url="/admin?msg=PDF uploaded successfully",
        status_code=303
    )

# =============================
# DELETE PDF
# =============================
@app.get("/delete/{filename}")
def delete_pdf(filename: str):

    file_path = os.path.join("uploads", filename)

    if os.path.exists(file_path):
        os.remove(file_path)

    return RedirectResponse(
        url="/admin?msg=PDF deleted successfully",
        status_code=303
    )

# =============================
# REBUILD VECTOR DATABASE
# =============================
@app.get("/rebuild")
def rebuild():

    global db

    if not os.path.exists("uploads"):
        return RedirectResponse(
            url="/admin?msg=Uploads folder not found",
            status_code=303
        )

    from langchain_community.document_loaders import PyPDFLoader

    documents = []

    for file in os.listdir("uploads"):
        if file.endswith(".pdf"):
            file_path = os.path.join("uploads", file)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print("Error loading file:", file, e)

    if not documents:
        return RedirectResponse(
            url="/admin?msg=No valid PDFs found",
            status_code=303
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    db = FAISS.from_documents(docs, get_embeddings())
    db.save_local("college_index")

    build_bm25()

    print("Vector DB rebuilt successfully.")

    return RedirectResponse(
        url="/admin?msg=Vector DB rebuilt successfully",
        status_code=303
    )

# =============================
# CHAT ENDPOINT
# =============================
@app.post("/chat")
def chat(q: Question):
    try:
        if db is None:
            return {"reply": "Vector database not built yet."}

        user_query = normalize_query(q.text)

        semantic_docs = db.similarity_search(user_query, k=5)

        keyword_docs = []
        if bm25:
            tokenized_query = user_query.split()
            scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
            keyword_docs = [documents_list[i] for i in top_indices]

        combined_docs = list({
            doc.page_content: doc for doc in (semantic_docs + keyword_docs)
        }.values())

        if not combined_docs:
            return {"reply": "Information not found."}

        context_parts = [doc.page_content for doc in combined_docs]
        context = " ".join(context_parts)

        prompt = f"""
You are an official AI academic assistant for GITAM University.

Answer strictly using the provided context.
If answer not found, reply exactly: Information not found.

Context:
{context}

Question:
{q.text}

Answer:
"""

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2
        )

        response = llm.invoke(prompt)
        answer = response.content.strip()

        return {"reply": answer}

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"reply": "Backend error occurred."}

# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)