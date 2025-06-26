import os
import argparse

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import speech_sentiment
import text_summarizer

from asr.dataset import get_featurizer
from asr.decoder import SpeechRecognitionEngine
from asr.engine import Recorder
from utils.helpers import get_assets_dir, get_models_dir


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize shared objects at startup
recorder = Recorder()               # Initialize recorder
featurizer = get_featurizer(16000)  # Initialize featurizer

asr_engine = None  # Initialize to None, to avoid issues during startup

# Serve static files (index.html)
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")

@app.route("/get-emotion", methods=["GET"])
def get_emotion():
    emotion = speech_sentiment.process_audio("temp/audio_temp.wav")
    
    return {"emotion" : emotion}

@app.route("/get-summary", methods=["GET"])
def get_summary():
    with open("temp/transcription.txt", "r") as f:
        transcript = f.read()
        summary = text_summarizer.summarize_texts(transcript)
        return {"summary" : summary}

@app.route("/transcribe/", methods=["POST"])
def transcribe_audio():
    """
    Transcribe an uploaded audio file using the preloaded ASR engine.
    """
    try:
        if asr_engine is None:
            return jsonify({"error": "ASR Engine is not initialized."}), 500

        recorded_audio = recorder.record()
        recorder.save(recorded_audio, "temp/audio_temp.wav")
        print("\nAudio recorded")

        # Use the preloaded ASR Engine to transcribe
        transcript = asr_engine.transcribe(asr_engine.model, featurizer, "temp/audio_temp.wav")

        print("\nTranscription:")
        print(transcript)

        with open("temp/transcription.txt", "w") as f:
            f.write(transcript)

        return jsonify({"transcription": transcript})

    except Exception as e:
        print(e)
        return jsonify({"error": f"Internal server error: {e}"}), 500


def main(args):
    global asr_engine
    print("Loading Speech Recognition Engine into cache...")
    try:
        asr_engine = SpeechRecognitionEngine(args.model_path, args.token_path) 
        print("ASR Engine loaded successfully.")
    except Exception as e:
        print(f"Error loading ASR Engine: {e}")
        asr_engine = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Demo: Record and Transcribe Audio")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the optimized ASR model.')
    parser.add_argument('--token_path', type=str, default=None, help='Path to the tokens file.')
    args = parser.parse_args()
    args.token_path = args.token_path if args.token_path else get_assets_dir() / "tokens.txt"
    args.model_path = args.model_path if args.model_path else get_models_dir("asr") / "asr_optimized_model.pt"

    main(args)
    
    app.run(host="127.0.0.1", port=8080, debug=True)