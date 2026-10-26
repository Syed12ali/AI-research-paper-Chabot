import re
import os
from typing import List, Dict, Tuple

import faiss
import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def normalize_text(text: str) -> str:
    """Clean up repeated whitespace and normalize simple text formatting."""
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pages_from_pdf(uploaded_file) -> List[Dict[str, object]]:
    """Read a PDF and return page text as a list of dictionaries."""
    temp_path = os.path.join("/tmp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    doc = fitz.open(temp_path)
    pages: List[Dict[str, object]] = []

    for page_number in range(doc.page_count):
        page = doc[page_number]
        text = page.get_text("text")
        cleaned = normalize_text(text)
        if cleaned:
            pages.append({"page_number": page_number + 1, "text": cleaned})

    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split large chunks into smaller overlapping segments."""
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        segment = text[start:end]

        if end < len(text):
            last_space = segment.rfind(" ")
            if last_space > max(0, chunk_size // 2):
                end = start + last_space
                segment = text[start:end]

        chunks.append(segment.strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return [chunk for chunk in chunks if chunk.strip()]


def build_index(pages: List[Dict[str, object]], model: SentenceTransformer):
    """Create text chunks and corresponding FAISS index."""
    chunks: List[Dict[str, object]] = []
    for page in pages:
        page_text = str(page["text"])
        for chunk_index, segment in enumerate(chunk_text(page_text)):
            chunks.append({
                "text": segment,
                "page_number": int(page["page_number"]),
                "chunk_index": chunk_index,
            })

    if not chunks:
        return [], None

    embeddings = model.encode([chunk["text"] for chunk in chunks], convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index


def lexical_overlap_score(text: str, question: str) -> int:
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q_tokens:
        return 0
    return len(q_tokens.intersection(t_tokens))


def retrieve_relevant_chunks(question: str, chunks: List[Dict[str, object]], index, model, top_k: int = 4):
    if not chunks or index is None:
        return []

    query_embedding = model.encode([question], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))

    matches = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        matches.append({
            "chunk": chunk,
            "distance": float(distance),
            "score": lexical_overlap_score(chunk["text"], question),
        })

    matches.sort(key=lambda item: (item["score"], -item["distance"]), reverse=True)
    return matches


def build_answer(question: str, matches: List[Dict[str, object]]) -> str:
    if not matches:
        return "I could not find relevant information in the uploaded paper. Please try a different question or upload a different PDF."

    primary = matches[0]["chunk"]
    text = primary["text"]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:2]) if len(sentences) > 1 else text[:400]
    return (
        f"{summary} This answer is supported by the uploaded paper, especially page {primary['page_number']}."
    )


st.set_page_config(page_title="AI Research Paper Q&A Chatbot", layout="wide")

st.title("📚 AI Research Paper Q&A Chatbot")
st.caption("Upload a PDF research paper and ask a question. Answers are grounded in the paper content and include page references.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting pages and building a search index..."):
        pages = extract_pages_from_pdf(uploaded_file)
        model = load_model()
        chunks, index = build_index(pages, model)

    if not chunks:
        st.warning("The PDF was uploaded, but no readable text could be extracted. Please try another file.")
        st.stop()

    st.success(f"Loaded {len(pages)} pages and {len(chunks)} text chunks.")

    question = st.text_input("Ask a question about the paper", placeholder="Example: What is the main contribution of this paper?")

    if question:
        matched_chunks = retrieve_relevant_chunks(question, chunks, index, model, top_k=4)
        answer = build_answer(question, matched_chunks)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Supporting passages")
        for item in matched_chunks[:3]:
            chunk = item["chunk"]
            st.markdown(f"**Page {chunk['page_number']}**")
            st.write(chunk["text"])
            st.markdown("---")
else:
    st.info("Please upload a PDF research paper to begin.")

st.markdown("\n")
st.markdown("Built with Python, Streamlit, SentenceTransformers, and FAISS.")
