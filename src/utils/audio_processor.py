import os
from typing import List, Dict, Any
from faster_whisper import WhisperModel

class AudioProcessor:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the Faster-Whisper model.
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribes an audio file and returns a list of segments with text and timestamps.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self.model.transcribe(audio_path, beam_size=5)

        results = []
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        return results

if __name__ == "__main__":
    # Quick test if run directly
    import sys
    if len(sys.argv) > 1:
        processor = AudioProcessor()
        file_path = sys.argv[1]
        try:
            print(f"Transcribing {file_path}...")
            transcription = processor.transcribe(file_path)
            for seg in transcription:
                print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python src/utils/audio_processor.py <path_to_audio_file>")
