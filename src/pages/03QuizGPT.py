import json
from langchain_openai import ChatOpenAI
import streamlit as st
from torch import Stream
from utils.document import format_docs
from utils.file import save_file
from utils.langchain import get_docs_from_file
from utils.streamlit import set_page_config
from langchain_community.retrievers import WikipediaRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from app.quiz.utils.fc import fc_quiz
from langchain.schema.runnable import RunnableLambda


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        return json.loads(text.replace("```", "").replace("json", ""))


output_parser = JsonOutputParser()

set_page_config("QuizGPT Home")

docs = None
topic = None

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        fc_quiz,
    ],
)
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 5 "questions" to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Context: {context}
""",
        )
    ]
)
question_chain = {"context": format_docs} | question_prompt | llm


def format_quiz(response):
    return response.additional_kwargs["function_call"]["arguments"]


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):

    chain = question_chain | RunnableLambda(format_quiz) | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)


choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article"))


if choice == "File":
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    if file:
        path = "./src/app/quiz"
        docs = get_docs_from_file(file, path)
else:
    topic = st.text_input("Enter a topic to search on Wikipedia")
    if topic:
        docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
        ### Welcome to QuizGPT
        
        I will make a quiz for you based on the file you upload or the topic you search on Wikipedia.

        Please upload a file or enter a topic to search on Wikipedia.
        """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select the correct answer",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Incorrect!")
        button = st.form_submit_button("Submit")
