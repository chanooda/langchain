import os
from dotenv import load_dotenv
from langchain_classic.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import streamlit as st
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter

load_dotenv()

translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following question into English. Output only the translation.\n\n{question}"
)

answers_prompt = ChatPromptTemplate.from_template("""
     Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
""")

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    # st.write(answers)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "data": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['data']}\n"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    header.decompose()
    footer.decompose()

    text = soup.get_text().replace("\n", " ")

    return text


def get_embeddings(api_key):
    # chunk_size: 요청당 텍스트 개수 (요청당 300k 토큰 제한 회피)
    if api_key:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
            chunk_size=200,
        )
    # 사이드바에 키가 없으면 .env의 OpenRouter 키로 fallback
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
        chunk_size=200,
    )


def get_llm(api_key):
    common = dict(
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    if api_key:
        return ChatOpenAI(api_key=api_key, model="gpt-4o-mini", **common)
    # 사이드바에 키가 없으면 .env의 OpenRouter 키로 fallback
    return ChatOpenRouter(model="openai/gpt-4o-mini", **common)


@st.cache_resource(show_spinner="Loading website...")
def load_website(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )

    loader = SitemapLoader(
        url,
        filter_urls=[r"^.*\/(ai-gateway|vectorize|workers-ai)\/.*"],
        parsing_function=parse_page,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(
        docs,
        get_embeddings(api_key),
    )
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
)


st.title("SiteGPT")

st.markdown("""
    Cloudflare에 대해서 뭐든지 물어보세요.
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
    """)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[GitHub 저장소](https://github.com/chanooda/langchain)")

if not api_key:
    st.caption("API Key를 입력하지 않으면 기본 LLM(OpenRouter)을 사용합니다.")

# get_answers, choose_answer에서 사용
llm = get_llm(api_key)

url = "https://developers.cloudflare.com/sitemap-0.xml"


try:
    retriever = load_website(url, api_key)
except Exception as e:
    st.error(f"웹사이트를 불러오지 못했습니다. API Key를 확인해주세요.\n\n{e}")
    st.stop()

if "processing" not in st.session_state:
    st.session_state.processing = False


def start_processing():
    st.session_state.processing = True
    # disabled 값이 바뀌면 위젯이 새로 만들어져 입력값이 사라지므로 미리 저장
    st.session_state.pending_query = st.session_state.query


if retriever:
    query = st.text_input(
        "질문 입력",
        key="query",
        on_change=start_processing,
        disabled=st.session_state.processing,
    )
    # 문서가 영어라서 검색에는 영어로 번역한 질문을 사용
    translate_chain = translate_prompt | llm | StrOutputParser()

    chain = (
        {
            "docs": translate_chain | retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )

    if st.session_state.processing:
        query = st.session_state.get("pending_query", "")
        try:
            if query:
                with st.spinner("답변을 찾는 중..."):
                    result = chain.invoke(query)
                st.session_state.answer = result.content
        except Exception as e:
            st.session_state.answer = f"에러가 발생했습니다: {e}"
        # 에러가 나도 입력창이 다시 활성화되도록 항상 해제 후 재실행
        st.session_state.processing = False
        st.rerun()

    if "answer" in st.session_state:
        st.write(st.session_state.answer)
