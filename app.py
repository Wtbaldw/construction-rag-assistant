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
