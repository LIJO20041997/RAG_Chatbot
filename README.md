## Retrieval-Augmented Generation (RAG) Chatbot
    This project implements a Retrieval Augmented Generation Chatbot using OpenAI (LLMs), FAISS vectorstore and Langchain to provide contextually relevant and accurate answers to use queries
## Architecture Overview
    The RAG chatbot uses a combination of document retrieval and generation to answer user questions. The following      high-level architecture describes the major components of the system:

    - Document Loader: Uses PyPDFLoader to load and extract text from PDF files.
    - Text Chunking: Splits document text into smaller chunks using RecursiveCharacterTextSplitter for efficient retrieval.
    - Vector Database (FAISS): Stores document chunks as embeddings using FAISS for similarity-based search.
    - Language Model (LLM): OpenAI’s GPT-based model processes user queries and generates answers based on the retrieved context.
    - Chat API Endpoint: Exposes a /questioning_pdf/ endpoint using FastAPI to interact with the chatbot.
## Key Components
    - LangChain: For document processing, embedding creation, and retrieval.
    - FAISS: For storing and retrieving document embeddings.
    - OpenAI API: Used to generate responses based on retrieved context.
    - FastAPI: Exposes the chatbot as an API for interaction.

## Architecture Diagram

                +-------------------+
                |    User Query     |
                +---------+---------+
                          |
                          v
                 +-----------------+
                 |   Chat API      |     (FastAPI Endpoint)
                 +-----------------+
                          |
                          v
               +----------------------+
               |   Document Loader    |
               |  (PDF Loader)        |
               +----------------------+
                          |
                          v
            +-----------------------------+
            | Text Chunking (Recursive    |
            | Character Text Splitter)    |
            +-----------------------------+
                          |
                          v
               +----------------------+
               |    Vector Store      |
               |      (FAISS)         |
               +----------------------+
                          |
                          v
           +-----------------------------+
           | Retrieval & Generation      |
           |   (RAG Chain with LLM)      |
           +-----------------------------+
                          |
                          v
               +----------------------+
               |  Response Generation |
               +----------------------+
                          |
                          v
                 +-------------------+
                 |  JSON Response    |
                 +-------------------+

