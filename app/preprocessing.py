# app/preprocessing.py

import re
import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class Preprocessor:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Cleans raw transcript text by:
        - Removing fillers, extra spaces
        - Sentence segmentation
        """
        # Remove common fillers
        fillers = ["um", "uh", "like", "you know", "basically", "so yeah"]
        for f in fillers:
            text = re.sub(rf"\b{f}\b", "", text, flags=re.IGNORECASE)

        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def segment_sentences(self, text: str) -> list:
        """
        Segments text into sentences using spaCy
        """
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
