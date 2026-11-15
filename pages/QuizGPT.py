import json
from langchain_classic.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_classic.schema import BaseOutputParser
from langchain_openrouter import ChatOpenRouter
from sqlalchemy import False_
import streamlit as st
import os
import streamlit as st
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

examples = [
    {"output": """```json
{{"question": "What is the color of the ocean?", "answers": [{{"answer": "Red", "correct": false}}, {{"answer": "Yellow", "correct": false}}, {{"answer": "Green", "correct": false}}, {{"answer": "Blue", "correct": true}}]}}
```"""},
    {"output": """```json
{{"question": "What is the capital or Georgia?", "answers": [{{"answer": "Baku", "correct": false}}, {{"answer": "Tbilisi", "correct": true}}, {{"answer": "Manila", "correct": false}}, {{"answer": "Beirut", "correct": false}}]}}
```"""},
    {"output": """```json
{{"question": "When was Avatar released?", "answers": [{{"answer": "2007", "correct": false}}, {{"answer": "2001", "correct": false}}, {{"answer": "2009", "correct": true}}, {{"answer": "1998", "correct": false}}]}}
```"""},
    {"output": """```json
{{"question": "Who was Julius Caesar?", "answers": [{{"answer": "A Roman Emperor", "correct": true}}, {{"answer": "Painter", "correct": false}}, {{"answer": "Actor", "correct": false}}, {{"answer": "Model", "correct": false}}]}}
```"""},
]


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)


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
        | JsonOutputParser()
    )
    quiz = chain.invoke(_docs)
    return quiz


llm = ChatOpenRouter(
    temperature=0.1,
    model="openai/gpt-4o-mini",
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
)


example_prompt = PromptTemplate.from_template("{output}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    The difficulty level of the quiz is: {level}.
    Adjust the difficulty of the questions according to this level:
    - Easy: ask about basic, explicitly stated facts. Wrong answers should be clearly different from the correct one.
    - Normal: ask about key details and require understanding of the text. Wrong answers should be plausible.
    - Hard: ask about subtle details and require connecting multiple parts of the text. Wrong answers should be very similar to the correct one.
    - Very Hard: ask questions requiring deep inference, analysis, and precise details. Wrong answers should be highly plausible and tricky.

    Each example below shows ONE question as a JSON object. Your output must be a single JSON array containing 10 such objects, inside one ```json code block. Do not output anything other than that code block.

    Question examples:
    """,
    suffix="""
            Your turn!

            Level: {level}
            Context: {context}
    """,
    input_variables=["context", "level"],
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

    def start_quiz():
        st.session_state["generating"] = True

    if st.session_state.get("generating"):
        try:
            with st.spinner("퀴즈 생성중..."):
                quiz = run_quiz_chain(docs, topic if topic else file.name, option)
            with st.form("question_form"):
                for question in quiz:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                    elif value is not None:
                        st.error("Wrong")

                button = st.form_submit_button()

        except json.JSONDecodeError:
            st.error("퀴즈를 JSON으로 읽지 못했어요. 다시 시도해 주세요.")
            st.session_state["generating"] = False

    else:
        option = st.selectbox(
            "난이도를 선택해주세요.", ("Easy", "Normal", "Hard", "Very Hard")
        )
        st.button("시작", on_click=start_quiz)
