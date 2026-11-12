import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

st.set_page_config(
    page_title="DocumentGPT",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenRouter(
    temperature=0.1,
    model="nvidia/nemotron-3.5-lightning:free",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


def format_answer(answers):
    return "\n\n".join(answer.page_content for answer in answers)


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message=message["message"], role=message["role"], save=False)


@st.cache_resource(show_spinner=True)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredLoader(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n"],
    )
    docs = loader.load_and_split(text_splitter=text_splitter)

    store = LocalFileStore(f"./.cache/embeddings/{file.name}")

    embedder = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
    )

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedder,
        store,
    )

    vector_store = InMemoryVectorStore.from_documents(docs, cached_embedder)
    retriever = vector_store.as_retriever()

    return retriever


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            오직 제공되는 Context을 참고해서 질문에 대답하세요. 모르면 모른다고 대답하세요. 절대 꾸며내지마세요.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown("너 사용해라 이 DocumentGPT. 파일 업로드 그리고 질문해")

file = st.file_uploader("파일을 업로드해주세요.", type=["pdf", "txt", "docx"])

if file:

    retriever = embed_file(file)

    send_message("준비됨. 질문 고", "ai", save=False)

    paint_history()

    message = st.chat_input("질문하라 당장")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_answer),
                "question": RunnablePassthrough(),
            }
            | template
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []
