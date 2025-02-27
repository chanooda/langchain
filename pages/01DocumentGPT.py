import streamlit as st
from utils.page_config import set_page_config

set_page_config("DocumentGPT Home")

if "messages" not in st.session_state:
    st.session_state.messages = []


for data in st.session_state.messages:
    [message, role] = data.values()
    with st.chat_message(role):
        st.write(message)


def send_message(message, role):

    with st.chat_message(role):
        st.write(message)

    st.session_state.messages.append({"message": message, "role": role})


message = st.chat_input("Send a message to the AI...")

if message:
    send_message(message, "human")
