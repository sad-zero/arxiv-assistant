import logging
import streamlit as st

from core.bootstrap import get_arxiv_searcher

logging.basicConfig(level=logging.DEBUG)

st.title("🔎 Arxiv Assistant")
"""
Arxiv Assistant helps you to search proper arxiv papers.

[Source Code](https://github.com/sad-zero/arxiv-assistant.git)
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'am arxiv assistant. I can search arxiv and answer your question. What can I do for you?",
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if query := st.chat_input(placeholder="What is prompt engineering?"):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    arxiv_searcher = get_arxiv_searcher()
    with st.chat_message("assistant"):
        response = arxiv_searcher.search(query)
        answer = f"{response.answer}\n\n"
        answer += "# Related Papers\n"
        for abstract in response.data:
            answer += f"- [{abstract.title}]({abstract.link})\n"
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.strip()}
        )
        st.markdown(answer)
