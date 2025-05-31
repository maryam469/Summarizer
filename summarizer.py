import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

##page settings
st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("Text Summarizer")
st.markdown("Enter your English paragraph below to get a short summary.")


@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model,tokenizer

model, tokenizer = load_model()

#input box
text = st.text_area("Enter English Text Here", height = 200)

##summarize
if st.button("Summarizer"):
    if text.strip() =="":
        st.warning("pLz enter some text to summarize.")
    else:
        with st.spinner("Summarizing...."):
            input_text ="summarize: " + text.strip()
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(input_ids, max_length = 100, min_length =30,length_penalty=2.0,num_beams=4,early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


            st.success("Summary Generated!")
            st.subheader("Summary")
            st.markdown(summary)


st.markdown("---")