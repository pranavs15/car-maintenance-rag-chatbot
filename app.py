import streamlit as st
import sys

sys.path.append("src")

from chatbot import ask_question

st.set_page_config(
    page_title="Car Maintenance Assistant",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Maintenance Assistant")

st.write("Ask any maintenance related question.")

question = st.text_input(
    "Enter your question",
    placeholder="Example: How do I replace engine oil?"
)

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching manuals..."):

            result = ask_question(question)

        st.success("Answer Generated")

        st.subheader("Answer")

        st.write(result["answer"])

        st.subheader("Retrieved Context")

        st.text_area(
            "",
            result["context"],
            height=350
        )