# AI Research Paper Q&A Chatbot

A complete Streamlit application that lets users upload a PDF research paper, extract its content, convert passages into embeddings, and answer questions using similarity search with page references.

## Features

- PDF upload in the browser
- Automatic text extraction from PDF pages
- Text chunking for efficient retrieval
- Embedding generation using SentenceTransformers
- FAISS vector search for semantic retrieval
- Question answering with evidence from supporting passages
- Page references included in the results
- Clean Streamlit UI

## Project structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.gitignore` - Standard ignores for Python projects
- `README.md` - Setup and usage documentation

## Installation

1. Clone this repository.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload a PDF, and ask a question.

## Example questions

- What is the main contribution of this paper?
- What problem does the paper address?
- What are the key findings or results?
- What methods were used in the experiments?

## Notes

This project uses a lightweight embedding model (`all-MiniLM-L6-v2`) and FAISS for similarity search. It is designed to work well for PDF research papers and can be extended with an LLM-based answer generator later if desired.
