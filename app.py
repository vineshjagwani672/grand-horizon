import os
from pathlib import Path

import streamlit as st


APP_TITLE = "Grand Horizon Hotel Knowledge Base Chatbot"
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "backend" / "data" / "grand_horizon_hotel_knowledge_base.pdf"


def load_streamlit_secrets():
    """Copy Streamlit Cloud secrets into environment variables used by backend/rag.py."""
    for key in ["GROQ_API_KEY", "GROQ_MODEL"]:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None

        if value and not os.getenv(key):
            os.environ[key] = str(value)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_qa_chain():
    from backend.rag import build_qa_chain

    return build_qa_chain()


st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
load_streamlit_secrets()

st.markdown(
    """
    <style>
        .stApp {
            background: #f7f2ea;
        }
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
        div[data-testid="stChatMessage"] {
            border-radius: 8px;
            padding: 8px;
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

if not PDF_PATH.exists():
    st.error("PDF file is missing. Add it at backend/data/grand_horizon_hotel_knowledge_base.pdf")
    st.stop()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud app secrets.")
    st.info(
        "In Streamlit Cloud, open App settings > Secrets and add: "
        'GROQ_API_KEY = "your_groq_api_key"'
    )
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

    try:
        chain = get_qa_chain()
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base..."):
                result = chain.invoke({"query": question})
                answer = result.get("result", "I could not generate an answer.")
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as exc:
        error_message = f"Error: {exc}"
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
