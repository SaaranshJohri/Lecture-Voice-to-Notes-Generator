# app/main.py

import os
import json
from datetime import datetime

from asr_module import ASRModule
from preprocessing import Preprocessor
from summarizer import Summarizer
from quiz_generator import QuizGenerator
from utils import clean_whitespace

# Create subfolders if not exist
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
SUBFOLDERS = ["transcripts", "summaries", "quizzes"]
for sf in SUBFOLDERS:
    os.makedirs(os.path.join(BASE_DATA_DIR, sf), exist_ok=True)


def pipeline(audio_file, num_questions=5, save=True):
    """
    Full pipeline: ASR -> Preprocessing -> Summarization -> Quiz
    Returns dict with transcript, summary, quiz.
    Also saves outputs in data/ if save=True.
    """

    # 1. Speech-to-Text
    asr = ASRModule(model_size="small")
    transcript = asr.transcribe(audio_file)

    # 2. Preprocessing
    preproc = Preprocessor()
    clean_text = preproc.clean_text(transcript)
    sentences = preproc.segment_sentences(clean_text)

    # 3. Summarization
    summarizer = Summarizer()
    summary = summarizer.summarize(" ".join(sentences))

    # 4. Quiz Generation
    qg = QuizGenerator()
    quiz = qg.generate_questions(summary, num_questions=num_questions)

    # Prepare results
    results = {
        "transcript": clean_whitespace(transcript),
        "summary": clean_whitespace(summary),
        "quiz": quiz
    }

    # ---------------- SAVE ----------------
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        run_id = f"{base_name}_{timestamp}"

        # Save transcript
        transcript_path = os.path.join(BASE_DATA_DIR, "transcripts", f"{run_id}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(results["transcript"])

        # Save summary
        summary_path = os.path.join(BASE_DATA_DIR, "summaries", f"{run_id}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(results["summary"])

        # Save quiz (JSON)
        quiz_path = os.path.join(BASE_DATA_DIR, "quizzes", f"{run_id}.json")
        with open(quiz_path, "w", encoding="utf-8") as f:
            json.dump(results["quiz"], f, indent=2, ensure_ascii=False)

        print(f"[Saved] Transcript → {transcript_path}")
        print(f"[Saved] Summary   → {summary_path}")
        print(f"[Saved] Quiz      → {quiz_path}")

    return results


