import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import anthropic

# Setup Streamlit
st.set_page_config(page_title="Local RAG Chatbot")

# Embedding model and possible llm's
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Hugging Face model setence transformer, vectorizes context
LLM_PROVIDER = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Claude"])

# API keys user input
# can be claude or openAI
if LLM_PROVIDER == "OpenAI":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    anthropic_api_key = st.sidebar.text_input("Claude API Key", type="password")

st.title("Local RAG Chatbot with FAISS + Claude/OpenAI")

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
question = st.text_input("Ask a question about your documents")

embedder = SentenceTransformer(EMBEDDING_MODEL)
index = None
all_chunks = []
chunk_to_doc = []

# Loads all context in file and returns
def load_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = file.read().decode("utf-8")
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


if uploaded_files:
    # loop through docs and collect chunks of text
    for file in uploaded_files:
        raw_text = load_file(file)
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
        chunk_to_doc.extend([file.name] * len(chunks))

    # converts each chunk to a np array which is the vector used for FAISS similarity search
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

# get most relavent context
def get_top_k_chunks(query, k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    # similarty search
    D, I = index.search(query_embedding, k)
    return [all_chunks[i] for i in I[0]]

def call_llm(prompt):
    if LLM_PROVIDER == "OpenAI":
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    else:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

if st.button("Get Answer") and question:
    if index is None:
        st.warning("Please upload documents first.")
    else:
        top_chunks = get_top_k_chunks(question)
        context = "\n\n".join(top_chunks)
        # create prompt with collected context which best fits the query
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        answer = call_llm(prompt)
        st.markdown("### Answer:")
        st.write(answer)
