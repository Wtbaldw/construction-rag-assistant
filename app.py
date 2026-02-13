import streamlit as st
from rag_chat import answer_question

st.set_page_config(page_title="Construction Q&A Assistant", page_icon="🧱")

st.title("🧱 Construction Q&A Assistant")
st.write("Ask a question about construction, remodeling, or general topics.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Thinking..."):
        answer = answer_question(question)
    st.markdown(answer)

"""This file creates the web interface for the Construction Q&A chatbot using Streamlit. 
It sets up the page layout, accepts user input through a text field, and calls the answer_question() function from rag_chat.py to generate responses.
 Streamlit is used to render the UI and display answers dynamically,
 allowing users to interact with the Retrieval-Augmented Generation (RAG) system through a simple, browser-based interface.
"""