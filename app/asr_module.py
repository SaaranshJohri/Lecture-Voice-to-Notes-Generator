# app/asr_module.py

import whisper

class ASRModule:
    def __init__(self, model_size="small"):
        """
        Initialize Whisper ASR model.
        Available sizes: tiny, base, small, medium, large
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes an audio file into text.
        
        Args:
            audio_path (str): Path to the audio file (mp3, wav, m4a supported)
        
        Returns:
            str: Transcribed text
        """
        # fp16=False is necessary for Windows/CPU
        result = self.model.transcribe(audio_path, fp16=False)
        return result["text"].strip()
