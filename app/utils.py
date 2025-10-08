# app/utils.py
import re

def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def split_into_chunks(text: str, max_len=500):
    words = text.split()
    chunks, chunk = [], []
    for word in words:
        if len(" ".join(chunk)) + len(word) < max_len:
            chunk.append(word)
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks
