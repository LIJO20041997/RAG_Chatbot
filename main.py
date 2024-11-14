from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI,HTTPException


from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "Documents/attention-is-all-you-need-Paper.pdf"

def get_pdf_text(PDF_PATH):
    pdf_reader = PyPDFLoader(PDF_PATH)
    documents = pdf_reader.load()
    return documents


def get_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=100)
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc.page_content)
        for split_text in split_texts:
            chunks.append({
                "page_content":split_text,
                "metadata":{"source":PDF_PATH, "page":doc.metadata['page']}
            })
    return chunks

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    texts = [chunk["page_content"]for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
    vectorstore.save_local("FAISS_DB")
    return vectorstore

def generate_response(vectorstore,question:str):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":7})
    llm = OpenAI(temperature=0.4, max_tokens=200)

    docs = retriever.invoke(question)

    system_prompt = (
        "You are an assistant for question-answering task. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know . keep the answer concise."
        "\n\n"
        "{context}"

    )
    prompt = ChatPromptTemplate.from_messages([('system', system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)

    responses = []
    for doc in docs:
        response = rag_chain.invoke({"input": question, "context": doc.page_content})
        responses.append({
            "answer": response['answer'],
            "source": doc.metadata.get("source","unkown source"),
            "page": doc.metadata.get("page","unknown page")
            })

    return {"response": responses}

app= FastAPI()

@app.post("/questioning_pdf/")
async def process_pdf_questioning(question:str=''):
    try:
        documents = get_pdf_text(PDF_PATH)
        chunks = get_chunks(documents)
        vectorstore = get_vectorstore(chunks)
        answer = generate_response(vectorstore, question)
        return {
            "result":answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))












