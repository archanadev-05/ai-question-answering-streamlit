import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

st.set_page_config(page_title="Question and Answering app")


model_name = "timpal0l/mdeberta-v3-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


@st.cache_resource
def load_model():

    return AutoModelForQuestionAnswering.from_pretrained(model_name)


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


inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = qa_model(**inputs)



if submit_btn:
    if not context.strip() or not question.strip():
        st.warning("⚠️ Please enter both context and question.")
    else:
        with st.spinner('Answering your question...'):
            start = outputs.start_logits.argmax().item()
            end   = outputs.end_logits.argmax().item() + 1

            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
            )



            st.success(answer)
            st.write(answer)   # shows the raw output — check if it's a dict or string
            # st.metric(label="Confidence", value=f"{round(result['score'] * 100, 2)}%")
# No else here — show nothing on initial load
import warnings
warnings.filterwarnings('ignore')

