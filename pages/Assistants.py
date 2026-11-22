from langchain_classic.document_loaders import WebBaseLoader
from langchain_classic.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
import streamlit as st

from openai import OpenAI

client = OpenAI()


def wikipedia_search(query: str) -> str:
    wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    return wiki.run(query)


def duckduckgo_search(query: str) -> str:
    ddg = DuckDuckGoSearchAPIWrapper()
    results = ddg.results(query, max_results=5)
    return "\n\n".join(f"{r['title']}\n{r['link']}\n{r['snippet']}" for r in results)


def scrape_website(url: str) -> str:
    docs = WebBaseLoader(url).load()
    text = "\n".join(
        line.strip() for line in docs[0].page_content.splitlines() if line.strip()
    )
    return text[:8000]


tools = [
    {
        "type": "function",
        "name": "wikipedia_search",
        "description": "Search Wikipedia for the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query for search",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "duckduckgo_search",
        "description": "Search DuckDuckGo. Returns title, link and snippet for each result.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query for search",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "scrape_website",
        "description": "Get the text content of a website.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "url for scrape",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

response = client.responses.create(model="gpt-6-luna", tools=tools)

st.set_page_config(
    page_title="Assistants GPT",
)
