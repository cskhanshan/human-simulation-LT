import whisper
import os

def transcribe_media(file_path):
    print("Current Directory:", os.getcwd())
    print("Looking for file:", file_path)

    if not os.path.exists(file_path):
        print("File not found!")
        return

    model = whisper.load_model("medium")  # Load the medium model for better accuracy

    print("Transcribing...")
    result = model.transcribe(file_path, language=None)  # Let it auto-detect Urdu + English

    print("\n Transcription Complete:\n")
    print(result["text"])

file_path = "whisper-file/whisper-2.ogg"
transcribe_media(file_path)
