from langchain.embeddings import CacheBackedEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
import streamlit as st


@st.cache_resource(show_spinner="...LOADING...")
def get_retriever__from_file(file):
    file_content = file.read()
    file_path = f"./src/app/document/files/{file.name}"
    cache_path_embeddings = f"./src/app/document/cache/embeddings/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

        loader = UnstructuredLoader(
            file_path=file_path,
        )
        docs = loader.load_and_split()
        # embedding
        cache_store = LocalFileStore(cache_path_embeddings)
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_store
        )

        # vector store
        vector_store = FAISS.from_documents(docs, cached_embeddings)

        # retriever
        retriever = vector_store.as_retriever()

        return retriever


@st.cache_resource(show_spinner="...LOADING...")
def get_chat_memory(file):
    memory = ConversationBufferMemory(
        memory_key="history", max_token_limit=300, return_messages=True
    )

    return memory
