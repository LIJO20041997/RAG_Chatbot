from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from fastapi import FastAPI,HTTPException,UploadFile,File
from io import BytesIO


from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "Documents/attention-is-all-you-need-Paper.pdf"

def get_pdf_text(PDF_PATH):
    texts = ""
    pdf_reader = PyPDFLoader(PDF_PATH)
    documents = pdf_reader.load()
    for docs in documents:
        texts+=docs.page_content
    return texts


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("FAISS_DB")
    return vectorstore

def generate_response(vectorstore,question:str):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":7})
    llm = OpenAI(temperature=0.4, max_tokens=200)

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

    response = rag_chain.invoke({"input": question})
    return response['answer']

app= FastAPI()

@app.post("/questioning_pdf/")
async def process_pdf_questioning(question:str=''):
    try:
        documents = get_pdf_text(PDF_PATH)
        chunks = get_chunks(documents)
        vectorstore = get_vectorstore(chunks)
        answer = generate_response(vectorstore, question)
        return{
            "status":True,
            "answer":answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))












