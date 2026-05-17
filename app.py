import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

st.title("RAG Chatbot")
st.caption("Upload a PDF and ask questions about it")

# Sidebar for uploading
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing your PDF..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # Load and chunk
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)
            
            # Build vector store
            embeddings = get_embeddings()
            db = FAISS.from_documents(chunks, embeddings)
            st.session_state.db = db
            st.session_state.messages = []
            os.unlink(tmp_path)
            st.success(f"✓ {uploaded_file.name} loaded! {len(chunks)} chunks indexed.")

    st.divider()
    if st.button("Use default (Attention paper)"):
        embeddings = get_embeddings()
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.session_state.db = db
        st.session_state.messages = []
        st.success("✓ Attention paper loaded!")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    if "db" not in st.session_state:
        st.warning("Please upload a PDF or click 'Use default' first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.db.as_retriever(
                        search_kwargs={"k": 3}
                    ).invoke(question)
                    context = "\n\n".join([d.page_content for d in docs])
                    prompt = f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
                    llm = get_llm()
                    response = llm.invoke(prompt)
                    answer = response.content
                    st.write(answer)
                    with st.expander("Sources"):
                        for doc in docs:
                            st.write(f"- Page {doc.metadata.get('page', '?')}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")