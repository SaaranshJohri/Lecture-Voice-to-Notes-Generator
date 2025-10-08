# app/quiz_generator.py

from transformers import pipeline
from keybert import KeyBERT

class QuizGenerator:
    def __init__(self, model_name="mrm8488/t5-base-finetuned-question-generation-ap"):
        """
        Initialize quiz generator with a fine-tuned T5 model + KeyBERT
        """
        # QG pipeline using a fine-tuned T5 model
        self.qg_pipeline = pipeline(
            task="text2text-generation",
            model=model_name,
            tokenizer=model_name,   # ensure SentencePiece model is included in HF cache
            device=-1,  
            framework="pt"             # CPU mode for Windows
        )
        # Keyword extraction
        self.kw_model = KeyBERT()

    def extract_keywords(self, text: str, num_keywords=5):
        """
        Extract top N keywords from text for generating questions
        """
        keywords = self.kw_model.extract_keywords(
            text,
            top_n=num_keywords,
            stop_words='english'
        )
        return [kw[0] for kw in keywords]

    def generate_questions(self, summary_text: str, num_questions=5):
        """
        Generate questions using keywords extracted from summary
        """
        keywords = self.extract_keywords(summary_text, num_keywords=num_questions)
        questions = []

        for kw in keywords:
            # Better prompt for question generation
            input_text = f"Context: {summary_text}\nAnswer: {kw}\nGenerate one question where the answer is the keyword."
            
            result = self.qg_pipeline(
                input_text,
                max_length=64,
                do_sample=False
            )
            question = result[0]["generated_text"]
            questions.append({"question": question, "answer": kw})

        return questions
