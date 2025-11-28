# app/services/qa.py
from typing import Tuple, List, Dict

# Later you can import and use sentence-transformers etc.
# from sentence_transformers import SentenceTransformer
# import numpy as np

# model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory mapping: doc_id -> list of chunks
DOC_CHUNKS: Dict[str, List[dict]] = {}


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """
    Naive text chunking by length/paragraphs.
    """
    chunks = []
    current = []
    current_len = 0

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue

        if current_len + len(para) > max_chars:
            chunks.append("\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n".join(current))

    return chunks


def process_document_text(doc_id: str, text: str) -> Tuple[str, dict]:
    """
    Prepare a document for Q&A:
      - chunk text
      - (later) compute embeddings
      - store in DOC_CHUNKS
      - return a short summary + meta
    """
    chunks = _chunk_text(text)
    DOC_CHUNKS[doc_id] = [
        {
            "chunk_id": i,
            "text": chunk,
            # "embedding": embedding_vector_here (later)
        }
        for i, chunk in enumerate(chunks)
    ]

    # Simple 'summary' for now: first few lines
    summary_lines = text.split("\n")[:5]
    summary = (
        "\n".join(summary_lines)[:500] if summary_lines else "Summary not available."
    )

    meta = {
        "num_chunks": len(chunks),
        # "doc_type": "unknown" (later you can use classifier)
    }

    return summary, meta


def answer_question(doc_id: str, question: str, meta: dict) -> Tuple[str, List[str]]:
    """
    For MVP:
      - Just return a dummy answer using first chunk as 'context'.
    Later:
      - Compute question embedding, do nearest neighbor search over chunks,
        then call LLM with those chunks to get answer.
    """
    chunks = DOC_CHUNKS.get(doc_id, [])
    if not chunks:
        return "I could not find any content for this document.", []

    # Dummy: use first chunk
    first_chunk = chunks[0]["text"]
    answer = (
        f'(Dummy answer) You asked: "{question}". '
        f"I would search in the document. Here's some context I see:\n\n"
        f"{first_chunk[:300]}..."
    )
    references = [f"Chunk 0: {first_chunk[:200]}..."]

    return answer, references
