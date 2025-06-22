import os
import numpy as np
import joblib
import argparse
import librosa
import torch

from sklearn.preprocessing import StandardScaler
from utils.helpers import get_models_dir
from speech_sentiment.model import HybridModel
from speech_sentiment.feature import getMELspectrogram, splitIntoChunks 


def process_audio(audio_file_path, sample_rate=16000):
    # Load and pad signal
    audio, _ = librosa.load(audio_file_path, sr=sample_rate, duration=3)
    signal = np.zeros(int(sample_rate * 3))
    signal[:len(audio)] = audio

    # Extract MEL spectrogram
    mel_spectrogram = getMELspectrogram(signal, sample_rate)
    chunks = splitIntoChunks(mel_spectrogram, win_size=128, stride=64)
    
    # Shape: (N, 128, 128) → (N, 1, 128, 128)
    chunks = np.expand_dims(chunks, axis=1)
    chunks_flat = chunks.reshape(chunks.shape[0], -1)
    chunks_scaled = scaler.transform(chunks_flat)
    chunks_scaled = chunks_scaled.reshape(-1, 1, 128, 128)

    # Convert to tensor
    chunks_tensor = torch.tensor(chunks_scaled).float().to(device).unsqueeze(0)  # shape: 1, N, 1, 128, 128

    # Predict
    with torch.inference_mode():
        model.eval()
        model.to(device)
        _, output_softmax, _ = model(chunks_tensor)
        predictions = torch.argmax(output_softmax, dim=1)
        predicted_emotion = EMOTIONS[predictions.item()]
        print(f"Predicted Emotion: {predicted_emotion}")
        return predicted_emotion

if __name__ == "__main__":
    model_path = get_models_dir("speech_sentiment/cnn_bilstm") / "speech_sentiment.pt"
    scaler_path = get_models_dir("speech_sentiment/cnn_bilstm") / "scaler.pkl"

    parser = argparse.ArgumentParser(description="Demoing the speech sentiment recognition engine.")
    parser.add_argument('--model_file', type=str, default=model_path, required=False, help='Path to SER model file.')
    parser.add_argument('--scaler_file', type=str, default=scaler_path, required=False, help='Path to SER model file.')
    parser.add_argument('--file_path', type=str, help='Input File', required=True)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.model_file, map_location=torch.device(device))  # or 'cpu' if needed
    emotions = checkpoint['emotions']
    EMOTIONS = {v: k for k, v in emotions.items()}
    model = HybridModel(len(EMOTIONS))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scaler = StandardScaler()
    scaler = joblib.load(args.scaler_file)
    if not os.path.exists(args.file_path):
        print(f"File {args.file_path} does not exist.")
        exit(1)
    process_audio(audio_file_path=args.file_path, sample_rate=16000)