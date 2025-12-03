# app/services/qa.py
import os
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# HF client
HF_TOKEN = os.environ.get("HF_TOKEN")
hf_client = None
if HF_TOKEN:
    hf_client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN,
    )

# Store chunks for each doc
DOC_CHUNKS: Dict[str, List[dict]] = {}

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------ chunking ------------------


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    chunks = []
    curr = []
    curr_len = 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if curr_len + len(line) > max_chars:
            chunks.append("\n".join(curr))
            curr = [line]
            curr_len = len(line)
        else:
            curr.append(line)
            curr_len += len(line)

    if curr:
        chunks.append("\n".join(curr))

    return chunks


# ------------------ document processing ------------------


def process_document_text(doc_id: str, text: str) -> Tuple[str, dict]:
    chunks = _chunk_text(text)

    embeddings = (
        embedding_model.encode(chunks, convert_to_tensor=False) if chunks else []
    )

    DOC_CHUNKS[doc_id] = [
        {
            "chunk_id": idx,
            "text": chunk,
            "embedding": np.asarray(emb, dtype="float32"),
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    # Summary = first chunk (simple)
    summary = (text[:500] + "...") if len(text) > 500 else text

    meta = {
        "num_chunks": len(chunks),
        "full_text": text,
    }
    return summary, meta


# ------------------ semantic search ------------------


def _retrieve_relevant_chunks(doc_id: str, question: str, top_k: int = 3) -> List[dict]:
    chunks = DOC_CHUNKS.get(doc_id, [])
    if not chunks:
        return []

    q_emb = embedding_model.encode([question], convert_to_tensor=False)[0]
    q_emb = np.asarray(q_emb, dtype="float32")

    scores = []
    for ch in chunks:
        emb = ch["embedding"]
        sim = float(
            np.dot(q_emb, emb) / ((np.linalg.norm(q_emb) * np.linalg.norm(emb)) + 1e-8)
        )
        scores.append((sim, ch))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scores[:top_k]]


# ------------------ HuggingFace QA ------------------


def _answer_with_hf(question: str, context: str) -> str:
    if not hf_client:
        return "HF_TOKEN not set. Please configure environment variable."

    try:
        result = hf_client.question_answering(
            model="deepset/roberta-base-squad2",
            question=question,
            context=context,
        )
        return result.get("answer", "No answer found.")
    except Exception as e:
        return f"Error calling HuggingFace API: {e}"


def answer_question(doc_id: str, question: str, meta: dict) -> Tuple[str, List[str]]:
    chunks = _retrieve_relevant_chunks(doc_id, question, top_k=3)

    if not chunks:
        return "No relevant content found.", []

    # Combine chunks for a strong context
    combined_context = "\n\n".join(ch["text"] for ch in chunks)

    answer = _answer_with_hf(question, combined_context)

    references = [f"Chunk {ch['chunk_id']}: {ch['text'][:200]}..." for ch in chunks]

    return answer, references
