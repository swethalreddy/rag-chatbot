# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions over PDF documents using LangChain, FAISS, and Groq LLaMA.

## Features
- Upload any PDF and chat with it instantly
- Uses FAISS vector store for fast semantic search
- Powered by Groq LLaMA 3.3 70B
- Built with Streamlit for a clean UI
- Shows source pages for every answer

## Tech Stack
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLaMA 3.3 70B
- Streamlit
- Python

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your GROQ_API_KEY to `.env`
4. Run: `streamlit run app.py`
