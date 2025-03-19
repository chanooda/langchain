import streamlit as st

from utils.langchain import get_chat_memory, get_retriever_from_file
from operator import itemgetter
from langchain.memory import chat_memory
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, response, *args, **kwargs):
        save_message("ai", self.message)

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def reset_chat_history():
    st.session_state["messages"] = []


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


def paint_document_layout(llm, embeddings, prompt, path):
    st.markdown(
        """ 
    Welcome!
            
    Use this chatbot to ask questions about your files!
    """
    )
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
        on_change=reset_chat_history,
    )

    if file:

        retriever = get_retriever_from_file(
            file,
            embeddings=embeddings,
            path=path,
        )
        chat_memory = get_chat_memory(file)

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
                | RunnablePassthrough.assign(
                    history=RunnableLambda(chat_memory.load_memory_variables)
                    | itemgetter("history")
                )
                | prompt
                | llm
            )

            with st.chat_message("ai"):
                response = chain.invoke(message)
                chat_memory.save_context(
                    {"input": message}, {"output": response.content}
                )

    else:
        reset_chat_history()
