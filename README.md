# Local RAG Chatbot (FAISS + Claude/OpenAI)

This is a locally hosted Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. It allows you to upload your own documents, ask natural language questions, and receive answers powered by OpenAI or Anthropic Claude.

## Features

- Upload `.pdf`, `.txt`, or `.md` files
- Automatically chunks and indexes document content with local embeddings
- Uses FAISS for fast vector similarity search
- Supports answering questions using Claude and OpenAI API

## Tools and Frameworks Used

- Streamlit – for the web UI
- sentence-transformers (`all-MiniLM-L6-v2`) – for local embeddings
- FAISS – local vector similarity search
- PyPDF2 – for PDF text extraction
- OpenAI & Anthropic SDKs – for calling LLMs

## How to Use

1. Clone the repo:

   ```
   git clone git@github.com:ouefsoupe/rag_client_project.git
   ```

2. Install dependencies:

   ```
   pip install streamlit faiss-cpu sentence-transformers PyPDF2 openai anthropic
   ```

3. Run the app:

   ```
   streamlit run app.py
   ```

4. Enter your OpenAI or Claude API key in the sidebar.

5. Upload documents and ask questions.
