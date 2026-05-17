from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def ingest_docs():
    documents = []
    docs_folder = "docs"
    
    for file in os.listdir(docs_folder):
        if file.endswith(".pdf"):
            print(f"Loading {file}...")
            loader = PyPDFLoader(os.path.join(docs_folder, file))
            documents.extend(loader.load())
    
    if not documents:
        print("No PDFs found in docs/ folder!")
        return
    
    print(f"Splitting {len(documents)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    print("Creating embeddings (this takes a minute)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("Saving to FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    
    print(f"Done! {len(chunks)} chunks saved to faiss_index/")

if __name__ == "__main__":
    ingest_docs()