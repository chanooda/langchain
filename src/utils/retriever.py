from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


def get_retriever__from_file(file):
    file_content = file.read()
    file_path = f"./src/app/document/files/{file.name}"
    cache_path_embeddings = f"./src/app/document/cache/embeddings/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

        # llm
        llm = ChatOpenAI(temperature=0.1)

        # document
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=600, chunk_overlap=100, separator="\n"
        )
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
