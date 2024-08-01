import streamlit as st
import os
import subprocess
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def create_vector_index(pdf_url):
    loader = PyPDFLoader(pdf_url)
    documents = loader.load()

    subprocess.run(['curl', '-O', pdf_url], capture_output=True, text=True)
    file_path = pdf_url.split('/')[-1]  # Extract filename from URL

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    docs = [Document(page_content=str(text)) for text in split_docs]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_db")
    vectorstore_disk = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vector_index = vectorstore_disk.as_retriever(search_kwargs={"k": 5})
    
    return vector_index

model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True)

def qa(vector_index):
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True)
    
    return qa_chain

def main():
    st.title("PDF Question Answering App")

    pdf_url = st.text_input("Enter the URL of the PDF document:")
    query = st.text_input("Ask a question about the document:")

    if pdf_url:
        vector_index = create_vector_index(pdf_url)
        
        if query:
            qa_chain = qa(vector_index)
            result = qa_chain({"query": query})
            st.write("Answer:", result["result"])

if __name__ == "__main__":
    main()