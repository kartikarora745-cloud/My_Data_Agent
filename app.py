import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.title("Spare Parts & Product Assistant (Live Google Sheet)")

# Streamlit Secrets se API Key
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY nahi mili! Streamlit Secrets check kar bhidusu.")
    st.stop()

# Tera Asli Google Sheet CSV Download URL
SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1-wznIijuD-qZgGoFxJcmZi-EKnWkiB6NQm7aVer24As/export?format=csv"

@st.cache_resource(ttl=300) # Har 5 minute me Live Sheet se naya data automatic load karega
def load_vectorstore():
    # Google Sheet se CSV download karke temp file me rakhta hai
    df = pd.read_csv(SHEET_CSV_URL)
    df.to_csv("temp_sheet.csv", index=False)
    
    loader = CSVLoader(file_path="temp_sheet.csv")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Tested aur Verified Active Model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="qwen/qwen3.8-27b"
)

template = """You are an assistant for question-answering tasks using latest product data.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_input = st.text_input("Apna sawal poocho:")

if user_input:
    response = rag_chain.invoke(user_input)
    st.write(response)