from langchain_classic.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.retrievers import WikipediaRetriever
import wikipedia
from datetime import timedelta

wikipedia.set_user_agent("LearnLangchainQuizGPT/0.1 (student learning project)")
wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=2))

st.set_page_config(
    page_title="QuizGPT",
)

st.title("QuizGPT")

docs = None
topic = None

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_doc(answers):
    return "\n\n".join(answer.page_content for answer in answers)


@st.cache_resource(show_spinner=True)
def split_file(file):
    file_content = file.read()
    os.makedirs("./.cache/quiz_files/files", exist_ok=True)
    os.makedirs("./.cache/quiz_files/embeddings", exist_ok=True)
    file_path = f"./.cache/quiz_files/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredLoader(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n"],
    )
    docs = loader.load_and_split(text_splitter=text_splitter)

    return docs


@st.cache_data(show_spinner="위키피디아 검색중...", ttl=3600)
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=3, lang="ko")
    return retriever.invoke(topic)


@st.cache_data(show_spinner=False)
def run_quiz_chain(_docs, topic, option):
    chain = (
        {"context": format_doc, "level": lambda _: option}
        | prompt
        | llm
        | JsonOutputKeyToolsParser(key_name="create_quiz", first_tool_only=True)
    )
    quiz = chain.invoke(_docs)
    return quiz


llm = ChatOpenRouter(
    temperature=0.1,
    model="openai/gpt-4o-mini",
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
).bind_tools(
    [function],
    tool_choice={"type": "function", "function": {"name": "create_quiz"}},
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    The difficulty level of the quiz is: {level}.
    Adjust the difficulty of the questions according to this level:
    - Easy: ask about basic, explicitly stated facts. Wrong answers should be clearly different from the correct one.
    - Normal: ask about key details and require understanding of the text. Wrong answers should be plausible.
    - Hard: ask about subtle details and require connecting multiple parts of the text. Wrong answers should be very similar to the correct one.
    - Very Hard: ask questions requiring deep inference, analysis, and precise details. Wrong answers should be highly plausible and tricky.

    Context: {context}
    """,
        )
    ]
)


select_option = st.selectbox("방식을 선택해주세요", ("File", "Wikipidia"))

if select_option == "File":
    file = st.file_uploader("파일을 업로드해주세요.", type=["pdf", "txt", "docx"])
    if file:
        docs = split_file(file)

else:
    topic = st.text_input("위키피디아 문서 검색")
    if topic:
        docs = wiki_search(topic)


if not docs:
    st.markdown("""
        QuizGPT에 오신 걸 환영합니다.

        위키피디아 문서 혹은 당신이 업로드 한 파일에 대해서 시험을 볼 수 있습니다.

        파일을 업로드 하거나, 위키피디아 검새을 해주세요.
    """)

else:

    option = None
    score = 0

    def start_quiz():
        st.session_state["generating"] = True

    if st.session_state.get("generating"):
        try:
            with st.spinner("퀴즈 생성중..."):
                quiz = run_quiz_chain(docs, topic if topic else file.name, option)
            with st.form("question_form"):
                for question in quiz["questions"]:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                        score += 1
                    elif value is not None:
                        st.error("Wrong")

                button = st.form_submit_button()
                if button:
                    if score == 10:
                        st.balloons
                    else:
                        st.write()

        except (KeyError, TypeError):
            st.error("퀴즈를 만들지 못했어요. 다시 시도해 주세요.")
            st.session_state["generating"] = False

    else:
        option = st.selectbox(
            "난이도를 선택해주세요.", ("Easy", "Normal", "Hard", "Very Hard")
        )
        st.button("시작", on_click=start_quiz)
