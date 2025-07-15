import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import retrieval_qa

# --- 1. INITIAL CONFIG ---
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    pass


# --- 2. CACHE LOGIC AND DATA LOAD
# Streamlit cache usage to avoid having to process the PDFs every time the app is loaded
@st.cache_resource
def load_and_proccess_data():
    """
    Load the PDFs, divide them in chunks. Then creates the embeddings.
    Those are saved in ChromaDB.
    This function is being executed only thanks to the cache 
    """

    loader = PyPDFDirectoryLoader("data/")
    documents = loader.load()

    #Split the documents in smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150 )
    texts = text_splitter.split_documents(documents)

    # Creates the embeddings using google model
    embeddings = GoogleGenerativeAI(model = "models/embedding-001" )

    # Creates a vector database ChromaDB and saves in a persistent directory
    # This secures that its not needed to recreate every time
    vector_store = Chroma.from_documents(texts, embeddings,persist_directory="./chroma_db" )

    return vector_store

# --- 3. PIPELINE CREATION FOR QUESTIONS AND ANSWETS
def create_qa_chain(vector_store):
    """
    Creates the pipeline (chain) of langchain to answer questions
    
    """
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # Creates the "retriever" that will search in the db vector
    retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # 'k': 5 recovers the 5 more relevant chunks

    # type of "chain". "stuff" injects all the recovered chunks recovered in the prompt 
    qa_chain = retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # optional to see which chunks were used
    )
    return qa_chain


    # --- 4. User interface with STREAMLIT ---
    st.set_page_config(page_title="BVG-GPT", page_icon="🚇")
    st.title("🚇 BVG-GPT: your public transport assistant in Berlin")

    # Load the data (this will use the cache if its being executed)
    try: 
        vector_store = load_and_proccess_data()
        # creates the chain each time the data is ready
        qa_chain = create_qa_chain(vector_store)

        # user input
        user_question = st.text_input("Write the question here: ", placeholder=" E.g. how much is the monthly ticket AB?")

        if user_question:
            with st.spinner("Searching on the documents and looking for an answer..."):
            # calls the chain with the user question
                result = qa_chain({"query": user_question})

                # shows the answer 
                st.success("Answer: ")
                st.write(result["result"])

                with st.expander("See utilized sources:  "):
                    for doc in result["source_documents"]:
                        st.info(f"Fuente: {doc.metadata['source']} (Página: {doc.metadata.get('page', 'N/A')})")
                        st.caption(doc.page_content)
                        



    except Exception as e:
        st.error(f"An error has occurred: {e}")
        st.warning("Make sure all the docs are in data folder and the credentials from google are in")
