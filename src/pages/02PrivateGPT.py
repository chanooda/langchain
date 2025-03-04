from operator import itemgetter
from langchain.memory import chat_memory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from utils.document import ChatCallbackHandler, paint_document_layout
from utils.streamlit import set_page_config
from utils.langchain import get_chat_memory, get_retriever__from_file


set_page_config("PrivateGPT Home")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

model = "mistral:latest"


llm = ChatOllama(
    model=model,
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context and not your training data. 
    If you don't know the answer, just say you don't know. DON'T make anything up'.
            
    Context: {context}
    Question: {question}
    """
)

paint_document_layout(
    llm=llm,
    embeddings=OllamaEmbeddings(model="nomic-embed-text"),
    prompt=prompt,
    path="./src/app/private",
)
