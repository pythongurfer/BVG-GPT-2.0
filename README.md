# Local-First RAG System for Document Analysis

This project implements a private, on-device Retrieval-Augmented Generation (RAG) system for querying and synthesizing information from a corpus of PDF documents. It addresses the critical need for factual, verifiable AI-powered analysis where data privacy and operational control are paramount. The entire system operates without reliance on external, third-party APIs, ensuring that all data remains within the local environment.

---

## Application Demonstration

The following demonstrates the system's interactive interface. A user query is submitted, and the system returns a synthesized answer grounded in the source documents, with full source attribution.

![Demo](assets/demo.gif)

<img src="assets/demo.gif" alt="Demo animation" width="600"/>

---

## Key Capabilities

- **Interactive Query Interface**  
  A clean, functional user interface built with Streamlit for real-time interaction.

- **Fully On-Device Architecture**  
  The entire data processing and inference pipeline runs locally, orchestrated by LangChain and powered by Ollama.

- **Guaranteed Data Privacy**  
  By design, source documents and user queries are never transmitted outside the local machine.

- **Verifiable, Citation-Based Generation**  
  All generated responses are synthesized directly from retrieved text segments, with citations provided to allow for verification and to mitigate model hallucination.

- **Persistent Vector Storage**  
  Leverages ChromaDB for efficient, on-disk storage of document embeddings, amortizing the cost of the initial data ingestion pipeline.

---

## System Architecture

The system is architected using LangChain to orchestrate a modular RAG pipeline, ensuring a clean separation of concerns between data processing, retrieval, and generation.

### Ingestion Pipeline (One-Time Execution)

- `PyPDFDirectoryLoader` loads all PDF files from the `./data/` directory.
- `RecursiveCharacterTextSplitter` intelligently divides the documents into smaller, overlapping text chunks, preserving semantic context.
- `OllamaEmbeddings` (utilizing the `nomic-embed-text` model) transforms each text chunk into a dense vector representation.
- `Chroma.from_documents` stores these chunks and their embeddings in a persistent on-disk vector database (`./chroma_db/`).

### Inference Pipeline (Real-Time Execution)

- The user's question is received via the Streamlit UI.
- The same `OllamaEmbeddings` model converts the user's query into a vector.
- `ChromaDB` performs a fast cosine similarity search to retrieve the *k* most relevant text chunks from the database.
- The `OllamaLLM` (running `phi3:mini`) receives a structured prompt containing the original question and the retrieved text chunks.
- The LLM synthesizes the provided context into a coherent, natural-language answer, which is then displayed in the UI.

---

## Technology Stack

- **Orchestration Framework:** LangChain  
- **Local Inference Server:** Ollama  
- **Language Model:** `phi3:mini`  
- **Embedding Model:** `nomic-embed-text`  
- **Vector Database:** ChromaDB  
- **Web UI Framework:** Streamlit  
- **PDF Parsing Library:** `pypdf`

---

## Local Deployment and Execution

### Prerequisites

- Python 3.10+
- An operational Ollama instance


Engineering Decisions and Rationale
Local-First vs. API-Based Architecture: The decision to utilize Ollama was strategic, prioritizing data privacy, operational autonomy, and cost control over the raw power of large-scale proprietary models. This architecture is ideal for sensitive data environments.

Model Selection and Resource Management: Initial prototyping with llama3:8b revealed significant inference latency on consumer-grade hardware. phi3:mini was selected to optimize the user experience by drastically reducing response times, accepting a minor trade-off in linguistic complexity for a major gain in performance. This highlights the critical balance between model capability and hardware constraints in applied AI.

Persistent Vector Storage: The use of ChromaDB's persist_directory is a crucial optimization. It decouples the computationally expensive ingestion pipeline from the main application loop, ensuring a fast and responsive startup time for the end-user after the initial data processing.

Semantic Chunking Strategy: The RecursiveCharacterTextSplitter was chosen to minimize semantic boundary fragmentation during the chunking process. By attempting to split along natural text structures (paragraphs, lines) before resorting to character-level splits, it produces more contextually coherent chunks, which directly improves the quality of context provided to the LLM.

Roadmap and Future Work
Advanced Retrieval Strategies: Investigate and implement more sophisticated retrieval mechanisms, such as HyDE (Hypothetical Document Embeddings) or a Multi-Query Retriever, to enhance retrieval accuracy for complex user queries.

Containerization: Package the application and its dependencies using Docker to ensure consistent, reproducible deployments across different environments.

Enhanced User Interface: Augment the Streamlit front-end with features such as conversation history, user feedback mechanisms, and the ability to select different source document sets.

Performance Benchmarking: Conduct formal benchmarking to evaluate the impact of GPU acceleration on LLM inference latency and explore further optimizations.