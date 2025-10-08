# app/summarizer.py

from transformers import pipeline

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize summarizer using HuggingFace pipeline
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length=200, min_length=50) -> str:
        """
        Generates a summary of the given text.
        """
        summary = self.summarizer(text, 
                                  max_length=max_length, 
                                  min_length=min_length, 
                                  do_sample=False)
        return summary[0]["summary_text"]
