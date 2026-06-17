import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "data" / "grand_horizon_hotel_knowledge_base.pdf"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_pdf_documents():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}.")
    return PyPDFLoader(str(PDF_PATH)).load()


def split_documents(documents):
    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    ).split_documents(documents)


def create_vectorstore():
    chunks = split_documents(load_pdf_documents())
    if not chunks:
        raise ValueError("No readable text found in PDF.")
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def load_vectorstore():
    if not (VECTORSTORE_DIR / "index.faiss").exists():
        return create_vectorstore()
    return FAISS.load_local(str(VECTORSTORE_DIR), get_embeddings(), allow_dangerous_deserialization=True)


def build_qa_chain():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to backend/.env.")

    vectorstore = load_vectorstore()

    prompt = PromptTemplate(
        template=(
            "You are Grand Horizon Hotel Knowledge Base Chatbot.\n"
            "Answer ONLY using the given context.\n"
            "If the answer is not in the context, say: "
            "'I could not find that information in the Grand Horizon Hotel knowledge base.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
