import json

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from openai import OpenAI
import streamlit as st

load_dotenv()

MODEL = "gpt-6-luna"
MAX_TOOL_ROUNDS = 10

INSTRUCTIONS = """
You are a research assistant.
Use the tools to gather information before answering, and cite the URLs you used.
Answer in the same language as the user's question.
"""


# ---------- 도구 함수 ----------
# 예외는 call_tool에서 문자열로 바꿔서 모델에 돌려준다. (앱이 멈추지 않고 모델이 다른 방법을 시도함)


def wikipedia_search(query: str) -> str:
    wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    return wiki.run(query)


def duckduckgo_search(query: str) -> str:
    ddg = DuckDuckGoSearchAPIWrapper()
    results = ddg.results(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['link']}\n{r['snippet']}" for r in results)


def scrape_website(url: str) -> str:
    docs = WebBaseLoader(url).load()
    if not docs:
        return "Empty page."
    text = "\n".join(
        line.strip() for line in docs[0].page_content.splitlines() if line.strip()
    )
    return text[:8000]


functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "scrape_website": scrape_website,
}


# ---------- 도구 스키마 ----------


def string_tool(name, description, param, param_desc):
    """문자열 파라미터 하나를 받는 strict function tool 스키마를 만든다."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {param: {"type": "string", "description": param_desc}},
            "required": [param],
            "additionalProperties": False,
        },
        "strict": True,
    }


tools = [
    string_tool(
        "wikipedia_search",
        "Search Wikipedia for background or encyclopedic information about a topic. "
        "Use for well-established facts, history and definitions.",
        "query",
        "Search keywords, e.g. 'XZ Utils'",
    ),
    string_tool(
        "duckduckgo_search",
        "Search the web with DuckDuckGo. Use for recent news or topics not covered by Wikipedia. "
        "Returns title, link and snippet for each result; "
        "call scrape_website on a link to read the full page.",
        "query",
        "Search keywords, e.g. 'XZ Utils backdoor timeline'",
    ),
    string_tool(
        "scrape_website",
        "Fetch the text content of a web page (truncated to 8000 characters). "
        "Use on URLs found via duckduckgo_search.",
        "url",
        "Full URL including https://",
    ),
]


# ---------- function call 루프 ----------


def call_tool(name: str, arguments: str) -> str:
    func = functions_map.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"
    try:
        return func(**json.loads(arguments))
    except Exception as e:
        return f"Error: {e}"


def run_agent(client: OpenAI, history: list) -> str:
    """모델이 도구 호출을 멈추고 최종 답변을 낼 때까지 반복한다.

    history는 Responses API의 input 아이템 리스트이며, 호출 결과가 그대로 누적된다.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            tools=tools,
            input=history,
        )
        # 모델 출력(reasoning, function_call, message)을 다음 요청에 그대로 넘긴다.
        history += response.output

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not function_calls:
            return response.output_text

        for call in function_calls:
            with st.status(f"🔧 {call.name}", expanded=False) as status:
                st.code(call.arguments, language="json")
                output = call_tool(call.name, call.arguments)
                st.text(output[:1000])
                status.update(state="complete")
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    return "도구 호출 횟수 제한에 도달했습니다. 질문을 좀 더 구체적으로 해주세요."


# ---------- UI ----------

st.set_page_config(
    page_title="Assistants",
)

st.title("Assistants")
st.markdown("Wikipedia, DuckDuckGo, 웹 스크래핑을 사용해서 리서치해 드립니다.")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[GitHub 저장소](https://github.com/chanooda/langchain)")
    if st.button("대화 초기화"):
        st.session_state.history = []
        st.session_state.messages = []

# api_key가 비어 있으면 .env의 OPENAI_API_KEY를 사용
client = OpenAI(api_key=api_key or None)

if "history" not in st.session_state:
    st.session_state.history = []  # 모델에 보내는 input 아이템 (도구 호출 포함)
if "messages" not in st.session_state:
    st.session_state.messages = []  # 화면에 보여줄 대화

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("무엇을 조사할까요?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = run_agent(client, st.session_state.history)
        except Exception as e:
            answer = f"에러가 발생했습니다: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
