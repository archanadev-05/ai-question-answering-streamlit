import streamlit as st
from transformers import pipeline
st.set_page_config(page_title="Question and Answering app")

@st.cache_resource
def load_model():

    return pipeline("question-answering", model='distilbert-base-cased-distilled-squad')


qa_model = load_model()

st.title("AI Powered Question and Answering WebApp")

col1, col2 = st.columns([2 , 1])

with col1:
    context = st.text_area("Text area for context",
                           height=150, placeholder="Enter Context here...")

    question = st.text_area("Text area for question",
                            height=150, placeholder="Ask your question here...")

    submit_btn = st.button("Ask question", type="primary")

with col2:
    st.markdown("    A Large Language Model–based QA bot that enables users to interact with AI through a simple and user-friendly interface.")

if context and question and submit_btn:
    with st.spinner('Answering your question...'):
        result = qa_model(question=question, context=context)
        st.success(result['answer'])
        st.metric(result['scores'])
else:
    st.markdown("Invalid Input...")

