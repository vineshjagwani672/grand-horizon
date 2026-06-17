import os
from pathlib import Path

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_TITLE = "Grand Horizon Hotel Knowledge Base Chatbot"
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "backend" / "data" / "grand_horizon_hotel_knowledge_base.pdf"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def read_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def chunk_text(text, chunk_size=1000, overlap=150):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap

    return chunks


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge_base():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "PDF file is missing. Add it at backend/data/grand_horizon_hotel_knowledge_base.pdf"
        )

    text = read_pdf_text(PDF_PATH)
    if not text.strip():
        raise ValueError("No readable text was found in the PDF.")

    chunks = chunk_text(text)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, matrix


def retrieve_context(question, chunks, vectorizer, matrix, top_k=4):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).flatten()
    best_indexes = scores.argsort()[-top_k:][::-1]
    return "\n\n".join(chunks[index] for index in best_indexes if scores[index] > 0)


def generate_answer(question, context):
    api_key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", DEFAULT_MODEL)

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit Cloud app secrets.")

    if not context.strip():
        return "I could not find that information in the Grand Horizon Hotel knowledge base."

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Grand Horizon Hotel Knowledge Base Chatbot. "
                    "Answer only from the provided context. If the answer is not in the context, "
                    "say: I could not find that information in the Grand Horizon Hotel knowledge base."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
    )
    return response.choices[0].message.content


st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .stApp { background: #f7f2ea; }
        .main-title {
            color: #284f3b;
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .subtitle {
            color: #6b5b4a;
            margin-bottom: 24px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='main-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Ask questions from the preloaded hotel PDF knowledge base.</div>",
    unsafe_allow_html=True,
)

if not get_secret("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud app secrets.")
    st.info(
        "In Streamlit Cloud, open App settings > Secrets and add: "
        'GROQ_API_KEY = "your_groq_api_key"'
    )
    st.stop()

try:
    chunks, vectorizer, matrix = load_knowledge_base()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything from the Grand Horizon Hotel knowledge base.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question from the PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                context = retrieve_context(question, chunks, vectorizer, matrix)
                answer = generate_answer(question, context)
                st.markdown(answer)
            except Exception as exc:
                answer = f"Error: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
