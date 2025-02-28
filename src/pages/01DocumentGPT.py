from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import CacheBackedEmbeddings
import streamlit as st
from utils.page_config import set_page_config
from utils.retriever import get_retriever__from_file

set_page_config("DocumentGPT Home")

st.markdown(
    """ 
    Welcome!
            
    Use this chatbot to ask questions about your files!
"""
)

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"])

if file:
    retriever = get_retriever__from_file(file)
