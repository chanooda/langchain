import streamlit as st


def set_page_config(page_title: str, page_icon: str = "🧊") -> None:
    """
    Set the page title and icon for the Streamlit app.

    Args:
    page_title (str): The title of the Streamlit app.
    page_icon (str): The icon of the Streamlit app.
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon)

    st.title(page_title)
