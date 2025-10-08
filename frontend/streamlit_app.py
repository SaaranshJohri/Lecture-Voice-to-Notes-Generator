# frontend/streamlit_app.py

import streamlit as st
import sys
import os

# Add app/ folder to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from asr_module import ASRModule
from preprocessing import Preprocessor
from summarizer import Summarizer
from quiz_generator import QuizGenerator

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lecture Voice-to-Notes Generator", layout="wide")
st.title("🎙️ Lecture Voice-to-Notes Generator")
st.markdown("Upload a lecture audio file and get **Transcript → Summary → Quiz** automatically.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")

# Model size for ASR
asr_model_size = st.sidebar.selectbox(
    "ASR Model Size",
    options=["small", "medium", "large"],
    index=0
)

# Number of quiz questions
num_quiz_questions = st.sidebar.slider(
    "Number of Quiz Questions",
    min_value=1, max_value=10, value=5, step=1
)

# Options to show/hide sections
show_transcript = st.sidebar.checkbox("Show Transcript", value=True)
show_summary = st.sidebar.checkbox("Show Summary", value=True)
show_quiz = st.sidebar.checkbox("Show Quiz", value=True)

# -----------------------------
# File uploader
# -----------------------------
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    st.success("✅ Audio uploaded successfully!")

    if st.button("Generate Notes & Quiz"):
        # 1. Speech-to-Text
        with st.spinner("Transcribing audio... ⏳"):
            asr = ASRModule(model_size=asr_model_size)
            transcript = asr.transcribe("temp_audio.mp3")

        if show_transcript:
            with st.expander("📜 Transcript", expanded=False):
                st.write(transcript)

        # 2. Preprocessing
        with st.spinner("Cleaning text..."):
            preproc = Preprocessor()
            clean_text = preproc.clean_text(transcript)
            sentences = preproc.segment_sentences(clean_text)

        # 3. Summarization
        with st.spinner("Summarizing notes..."):
            summarizer = Summarizer()
            summary = summarizer.summarize(" ".join(sentences))

        if show_summary:
            st.subheader("📝 Summary Notes")
            st.write(summary)

        # 4. Quiz Generation
        if show_quiz:
            with st.spinner("Generating quiz..."):
                qg = QuizGenerator()
                quiz = qg.generate_questions(summary, num_questions=num_quiz_questions)

            st.subheader("❓ Generated Quiz")
            for i, q in enumerate(quiz, 1):
                st.markdown(f"**Q{i}: {q['question']}**")
                st.markdown(f"*Answer:* {q['answer']}")

        st.success("🎉 Done! Notes and Quiz generated.")

        # Download summary
        st.download_button(
            label="📄 Download Notes as TXT",
            data=summary,
            file_name="lecture_notes.txt",
            mime="text/plain"
        )
