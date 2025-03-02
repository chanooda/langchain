from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from utils.streamlit import set_page_config
from utils.retriever import get_retriever__from_file


if "messages" not in st.session_state:
    st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, response, *args, **kwargs):
        save_message("ai", self.message)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


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
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def save_message(role, message):
    st.session_state["messages"].append({"role": role, "message": message})


def send_message(role, message, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(role, message)


def paint_messages():
    for message in st.session_state["messages"]:
        send_message(message["role"], message["message"], save=False)


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

    send_message("ai", "I am ready to answer your questions!", save=False)
    paint_messages()

    message = st.chat_input("Ask a question")

    if message:
        send_message("user", message)
        docs = retriever.invoke(message)

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state["messages"] = []
