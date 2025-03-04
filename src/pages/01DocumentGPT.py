from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
from utils.document import ChatCallbackHandler, paint_document_layout
from utils.streamlit import set_page_config


set_page_config("DocumentGPT Home")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer, just say you don't know. DON'T make anything up'.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

paint_document_layout(
    llm=llm, embeddings=OpenAIEmbeddings(), prompt=prompt, path="./src/app/document"
)
