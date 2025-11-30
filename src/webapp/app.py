import os
import argparse
import psutil
import time
from flask import Flask, request, jsonify, send_from_directory
import torch
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

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Initialize shared objects at startup
recorder = Recorder()    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
featurizer = get_featurizer(sample_rate=16000, device=device)
asr_engine = None

# Resource monitoring
process = psutil.Process()
resource_stats = {
    "asr_load": {},
    "transcription_sessions": []
}

def get_resource_usage():
    """Get current CPU and memory usage"""
    cpu_percent = process.cpu_percent(interval=0.1)
    mem_info = process.memory_info()
    
    return {
        "cpu_percent": cpu_percent,
        "ram_mb": mem_info.rss / (1024 * 1024),  # Convert to MB
        "ram_gb": mem_info.rss / (1024 * 1024 * 1024),  # Convert to GB
        "cpu_count": psutil.cpu_count(),
        "timestamp": time.time()
    }

def log_resources(stage, extra_info=None):
    """Log resource usage at different stages"""
    stats = get_resource_usage()
    stats["stage"] = stage
    if extra_info:
        stats.update(extra_info)
    
    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"CPU Usage: {stats['cpu_percent']:.2f}%")
    print(f"RAM Usage: {stats['ram_mb']:.2f} MB ({stats['ram_gb']:.3f} GB)")
    if extra_info and "duration" in extra_info:
        print(f"Duration: {extra_info['duration']:.2f}s")
    print(f"{'='*60}\n")
    
    return stats

# Serve static files (index.html)
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")

@app.route("/get-emotion", methods=["GET"])
def get_emotion():
    start_time = time.time()
    start_stats = get_resource_usage()
    
    emotion = speech_sentiment.process_audio("temp/audio_temp.wav")
    
    end_stats = get_resource_usage()
    duration = time.time() - start_time
    
    log_resources("Emotion Analysis", {
        "duration": duration,
        "ram_delta_mb": end_stats["ram_mb"] - start_stats["ram_mb"]
    })
    
    return {"emotion": emotion}

@app.route("/get-summary", methods=["GET"])
def get_summary():
    start_time = time.time()
    start_stats = get_resource_usage()
    
    with open("temp/transcription.txt", "r") as f:
        transcript = f.read()
    summary = text_summarizer.summarize_texts(transcript)
    
    end_stats = get_resource_usage()
    duration = time.time() - start_time
    
    log_resources("Text Summarization", {
        "duration": duration,
        "ram_delta_mb": end_stats["ram_mb"] - start_stats["ram_mb"]
    })
    
    return {"summary": summary}

@app.route("/transcribe/", methods=["POST"])
def transcribe_audio():
    """
    Transcribe an uploaded audio file using the preloaded ASR engine.
    """
    try:
        if asr_engine is None:
            return jsonify({"error": "ASR Engine is not initialized."}), 500
        
        # Start monitoring
        session_start = time.time()
        start_stats = get_resource_usage()
        
        # Record audio
        record_start = time.time()
        recorded_audio = recorder.record()
        recorder.save(recorded_audio, "temp/audio_temp.wav")
        record_duration = time.time() - record_start
        record_stats = get_resource_usage()
        
        print("\nAudio recorded")
        log_resources("Audio Recording", {
            "duration": record_duration,
            "ram_delta_mb": record_stats["ram_mb"] - start_stats["ram_mb"]
        })
        
        # Transcribe
        transcribe_start = time.time()
        transcript = asr_engine.transcribe(featurizer, "temp/audio_temp.wav")
        transcribe_duration = time.time() - transcribe_start
        transcribe_stats = get_resource_usage()
        
        print("\nTranscription:")
        print(transcript)
        
        log_resources("ASR Transcription", {
            "duration": transcribe_duration,
            "ram_delta_mb": transcribe_stats["ram_mb"] - record_stats["ram_mb"],
            "total_ram_delta_mb": transcribe_stats["ram_mb"] - start_stats["ram_mb"]
        })
        
        # Save session stats
        session_stats = {
            "session_duration": time.time() - session_start,
            "record_duration": record_duration,
            "transcribe_duration": transcribe_duration,
            "start_ram_mb": start_stats["ram_mb"],
            "end_ram_mb": transcribe_stats["ram_mb"],
            "ram_delta_mb": transcribe_stats["ram_mb"] - start_stats["ram_mb"],
            "peak_cpu_percent": transcribe_stats["cpu_percent"],
            "timestamp": time.time()
        }
        resource_stats["transcription_sessions"].append(session_stats)
        
        with open("temp/transcription.txt", "w") as f:
            f.write(transcript)
        
        return jsonify({
            "transcription": transcript,
            "resource_usage": session_stats
        })
        
    except Exception as e:
        print(e)
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route("/resource-stats", methods=["GET"])
def get_resource_stats():
    """Get detailed resource statistics"""
    current_stats = get_resource_usage()
    
    # Calculate averages if we have session data
    sessions = resource_stats["transcription_sessions"]
    avg_stats = {}
    
    if sessions:
        avg_stats = {
            "avg_transcribe_duration": sum(s["transcribe_duration"] for s in sessions) / len(sessions),
            "avg_ram_delta_mb": sum(s["ram_delta_mb"] for s in sessions) / len(sessions),
            "max_ram_delta_mb": max(s["ram_delta_mb"] for s in sessions),
            "total_sessions": len(sessions)
        }
    
    return jsonify({
        "current": current_stats,
        "asr_load_stats": resource_stats["asr_load"],
        "session_averages": avg_stats,
        "recent_sessions": sessions[-10:]  # Last 10 sessions
    })

def main(args):
    global asr_engine
    
    print("Loading Speech Recognition Engine into cache...")
    
    # Monitor ASR engine loading
    load_start = time.time()
    start_stats = get_resource_usage()
    
    try:
        asr_engine = SpeechRecognitionEngine(args.model_path, args.token_path)
        
        load_duration = time.time() - load_start
        end_stats = get_resource_usage()
        
        # Store ASR load stats
        resource_stats["asr_load"] = {
            "duration": load_duration,
            "ram_before_mb": start_stats["ram_mb"],
            "ram_after_mb": end_stats["ram_mb"],
            "ram_delta_mb": end_stats["ram_mb"] - start_stats["ram_mb"],
            "cpu_percent": end_stats["cpu_percent"]
        }
        
        print("ASR Engine loaded successfully.")
        log_resources("ASR Engine Loading", {
            "duration": load_duration,
            "ram_delta_mb": end_stats["ram_mb"] - start_stats["ram_mb"]
        })
        
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