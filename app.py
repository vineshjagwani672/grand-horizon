import os
from pathlib import Path

import streamlit as st
from groq import Groq
from pypdf import PdfReader


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

    return chunk_text(text)


def tokenize(text):
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what",
        "when", "where", "who", "why", "with", "how",
    }
    words = []
    for word in text.lower().split():
        clean_word = "".join(char for char in word if char.isalnum())
        if clean_word and clean_word not in stop_words:
            words.append(clean_word)
    return words


def retrieve_context(question, chunks, top_k=4):
    question_words = set(tokenize(question))
    scored_chunks = []

    for chunk in chunks:
        chunk_words = tokenize(chunk)
        if not chunk_words:
            continue

        score = sum(1 for word in chunk_words if word in question_words)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(chunk for _, chunk in scored_chunks[:top_k])


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
    chunks = load_knowledge_base()
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
                context = retrieve_context(question, chunks)
                answer = generate_answer(question, context)
                st.markdown(answer)
            except Exception as exc:
                answer = f"Error: {exc}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
