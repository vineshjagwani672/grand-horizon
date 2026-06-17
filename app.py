import os
import html
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

    return text, chunk_text(text)


def is_greeting(message):
    greetings = {"hi", "hello", "hey", "salam", "assalamualaikum", "assalam o alaikum"}
    return message.strip().lower() in greetings


def is_too_short(message):
    return len(tokenize(message)) == 0


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

    if not question_words:
        return ""

    for chunk in chunks:
        chunk_words = set(tokenize(chunk))
        if not chunk_words:
            continue

        score = len(question_words.intersection(chunk_words))
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


def answer_question(question, chunks):
    if is_greeting(question):
        return (
            "Hello! I am ready to help. Ask me about the Grand Horizon Hotel knowledge base, "
            "such as rooms, services, policies, dining, reservations, or hotel facilities."
        )

    if is_too_short(question):
        return "Please ask a complete question about the Grand Horizon Hotel knowledge base."

    context = retrieve_context(question, chunks)
    return generate_answer(question, context)


def render_message(role, content):
    safe_content = html.escape(content).replace("\n", "<br>")
    row_class = "user-row" if role == "user" else "assistant-row"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    label = "You" if role == "user" else "Grand Horizon Assistant"

    st.markdown(
        f"""
        <div class="message-row {row_class}">
            <div class="message-stack">
                <div class="message-label">{label}</div>
                <div class="message-bubble {bubble_class}">{safe_content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(199, 160, 89, 0.24), transparent 34%),
                linear-gradient(135deg, #111827 0%, #1f2933 52%, #284f3b 100%);
            color: #f8fafc;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        [data-testid="block-container"] {
            max-width: 900px;
            padding-top: 42px;
            padding-bottom: 120px;
        }
        .hero-panel {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(244, 200, 123, 0.35);
            border-radius: 18px;
            padding: 26px 28px;
            margin-bottom: 22px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(14px);
        }
        .main-title {
            color: #fff8e8;
            font-size: 36px;
            font-weight: 800;
            line-height: 1.14;
            margin-bottom: 8px;
        }
        .subtitle {
            color: #f4c87b;
            font-size: 16px;
            margin-bottom: 0;
        }
        [data-testid="stCaptionContainer"] {
            color: #d6e4d2;
            margin-bottom: 18px;
        }
        .message-row {
            display: flex;
            margin: 14px 0;
            width: 100%;
        }
        .assistant-row {
            justify-content: flex-start;
        }
        .user-row {
            justify-content: flex-end;
        }
        .message-stack {
            max-width: min(78%, 680px);
        }
        .message-label {
            color: #f4c87b;
            font-size: 12px;
            font-weight: 700;
            margin: 0 0 5px 2px;
        }
        .user-row .message-label {
            text-align: right;
            margin-right: 2px;
        }
        .message-bubble {
            border-radius: 14px;
            padding: 14px 16px;
            line-height: 1.6;
            font-size: 15px;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
            overflow-wrap: anywhere;
        }
        .assistant-bubble {
            background: #fffaf0;
            color: #17202a;
            border: 1px solid #f4c87b;
            border-bottom-left-radius: 4px;
        }
        .user-bubble {
            background: linear-gradient(135deg, #b54f37, #7f2f22);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-bottom-right-radius: 4px;
        }
        [data-testid="stChatInput"] {
            background: rgba(17, 24, 39, 0.72);
            border-top: 1px solid rgba(244, 200, 123, 0.28);
            padding: 16px 0;
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus {
            background: #ffffff !important;
            color: #111827 !important;
            border: 2px solid #f4c87b !important;
            border-radius: 14px !important;
            caret-color: #b54f37 !important;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #6b7280 !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] button {
            background: #f4c87b !important;
            color: #111827 !important;
            border-radius: 12px !important;
        }
        @media (max-width: 640px) {
            [data-testid="block-container"] {
                padding-left: 16px;
                padding-right: 16px;
            }
            .main-title {
                font-size: 28px;
            }
            .message-stack {
                max-width: 92%;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-panel">
        <div class="main-title">{APP_TITLE}</div>
        <div class="subtitle">Ask questions from the preloaded hotel PDF knowledge base.</div>
    </div>
    """,
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
    full_text, chunks = load_knowledge_base()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Knowledge base loaded: {len(chunks)} chunks from {PDF_PATH.name}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything from the Grand Horizon Hotel knowledge base.",
        }
    ]

for message in st.session_state.messages:
    render_message(message["role"], message["content"])

question = st.chat_input("Ask a question from the PDF...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    with st.spinner("Searching the knowledge base..."):
        try:
            answer = answer_question(question, chunks)
        except Exception as exc:
            answer = f"Error: {exc}"

    render_message("assistant", answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
