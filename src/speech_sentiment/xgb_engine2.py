import numpy as np
import joblib
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import get_models_dir

_model = None
_scaler = None
_label_encoder = None
_config = None

def load_model():
    """Load XGBoost model, scaler, and label encoder"""
    global _model, _scaler, _label_encoder, _config
    
    if _model is None:
        model_path = get_models_dir("speech_sentiment/xgb")
        
        # Load model components
        _model = joblib.load(model_path / "model.pkl")
        _scaler = joblib.load(model_path / "scaler.pkl")
        _label_encoder = joblib.load(model_path / "label_encoder.pkl")
        
        # Load configuration (with default fallback)
        try:
            import json
            config_path = model_path / "metrics" / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    _config = json.load(f)
            else:
                _config = {'n_mfcc': 13}
        except:
            _config = {'n_mfcc': 13}
    
    return _model, _scaler, _label_encoder, _config

def extract_features(file_path, config):
    """Extract audio features from a single file
    
    Args:
        file_path: Path to the audio file
        config: Configuration dictionary containing feature extraction parameters
        
    Returns:
        Dictionary of extracted features
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config['n_mfcc']), axis=1)
    
    # Extract Chroma features
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    
    # Extract Spectral Contrast features
    contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    
    # Combine features into dictionary
    features = {}
    
    # MFCC features
    for i in range(config['n_mfcc']):
        features[f'mfccs_mean_{i}'] = mfccs_mean[i]
    
    # Chroma features
    for i in range(12):
        features[f'chroma_mean_{i}'] = chroma_mean[i]
    
    # Contrast features
    for i in range(7):
        features[f'contrast_mean_{i}'] = contrast_mean[i]
    
    return features

def process_audio(audio_file_path):
    """Process audio file and return predicted emotion
    
    Args:
        audio_file_path: Path to the audio file to process
        
    Returns:
        str: Predicted emotion label
    """
    # Load models
    model, scaler, label_encoder, config = load_model()
    
    # Extract features from audio file
    features = extract_features(audio_file_path, config)
    
    # Convert features to DataFrame for consistent column ordering
    features_df = pd.DataFrame([features])
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    predicted_emotion = label_encoder.inverse_transform([prediction])[0]
    
    print(f"Predicted Emotion: {predicted_emotion}")
    return predicted_emotion