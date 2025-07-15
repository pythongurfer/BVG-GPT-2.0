# app_ollama.py

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 👇 1. IMPORTACIONES ACTUALIZADAS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA

# --- LÓGICA DE CACHÉ Y CARGA DE DATOS ---
@st.cache_resource
def load_and_process_data():
    st.info("1. Loading docs...")
    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    
    st.info("2. Dividing docs and creating embeddings (this can take time)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    st.info("3. Vector Database ready!")
    return vector_store

# --- CREACIÓN DEL PIPELINE ---
def create_qa_chain(vector_store):
    # 👇 2. USANDO EL MODELO LIGERO Y LA NUEVA CLASE OllamaLLM
    llm = OllamaLLM(model="phi3:mini")
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- INTERFAZ DE USUARIO ---
st.set_page_config(page_title="BVG-GPT", page_icon="🚇")
st.title("🚇 BVG-GPT: Your local transport Assistant in Berlin")
st.info("Make a question about fees, rules or conditions of BVG.")

try:
    vector_store = load_and_process_data()
    qa_chain = create_qa_chain(vector_store)

    user_question = st.text_input("Write your question here:", placeholder="Example: Can I take the dog on the train?")

    if user_question:
        with st.spinner("Searching and generating answer..."):
            result = qa_chain.invoke({"query": user_question}) # .invoke es el método actualizado en LangChain
            st.success("Answer:")
            st.write(result["result"])

            with st.expander("See sources used"):
                for doc in result["source_documents"]:
                    st.info(f"Source: {doc.metadata['source']} (Page: {doc.metadata.get('page', 'N/A')})")
                    st.caption(doc.page_content)

except Exception as e:
    st.error(f"Ha ocurrido un error: {e}")
    st.warning("Make sure that documents are located on 'data' folder and Ollama is running.")