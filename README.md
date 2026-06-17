# Grand Horizon Hotel Knowledge Base Chatbot

A simple course-project RAG chatbot built with:

- Streamlit deploy app
- React + Vite frontend
- FastAPI backend
- LangChain
- FAISS vector database
- Groq LLM
- PDF knowledge base loaded from `backend/data`

The UI does not allow PDF uploads. The backend automatically uses this file:

```text
backend/data/grand_horizon_hotel_knowledge_base.pdf
```

## Project Structure

```text
project-root/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── data/
│   │   └── grand_horizon_hotel_knowledge_base.pdf
│   ├── vectorstore/
│   ├── app.py
│   ├── ingest.py
│   ├── rag.py
│   ├── requirements.txt
│   ├── .env
│   └── .env.example
│
└── README.md
```

## Streamlit Cloud Deployment

This project is now ready to deploy on Streamlit Community Cloud using the root `app.py`.

Before deploying, make sure these files are pushed to GitHub:

```text
app.py
requirements.txt
runtime.txt
backend/rag.py
backend/data/grand_horizon_hotel_knowledge_base.pdf
```

In Streamlit Cloud:

1. Click **New app**.
2. Select your GitHub repository.
3. Set the main file path to:

```text
app.py
```

4. Open **Advanced settings > Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

5. Deploy the app.

Do not upload `.env`, `backend/venv`, `frontend/node_modules`, or local cache files.

## Important

Place your PDF file here before running the project:

```text
backend/data/grand_horizon_hotel_knowledge_base.pdf
```

## Backend Setup

Open a terminal in the project root.

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key in `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
FRONTEND_URL=http://localhost:5173
```

Create the FAISS vector database:

```bash
python ingest.py
```

Start the backend:

```bash
uvicorn app:app --reload
```

Backend URL:

```text
http://localhost:8000
```

Chat API endpoint:

```text
POST http://localhost:8000/chat
```

Example request:

```json
{
  "question": "What are the restaurant opening hours?"
}
```

Example response:

```json
{
  "answer": "The answer from the PDF will appear here."
}
```

## Frontend Setup

Open a second terminal in the project root.

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## How It Works

1. `ingest.py` reads the PDF from `backend/data`.
2. The PDF text is split into smaller chunks.
3. Gemini embeddings are created for each chunk.
4. FAISS stores the vectors in `backend/vectorstore`.
5. `app.py` loads the vector database.
6. The React app sends questions to `POST /chat`.
7. The backend retrieves matching PDF chunks and generates an answer.

## Notes

- The chatbot is instructed to answer only from the PDF content.
- If the answer is not found in the PDF, it will say it could not find that information.
- There is no PDF upload option in the frontend.
- The vector store is also created automatically on backend startup if it does not already exist and the PDF is available.
