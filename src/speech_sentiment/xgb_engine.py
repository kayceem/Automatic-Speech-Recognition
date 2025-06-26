#!/usr/bin/env python3
"""
XGBoost Inference Engine for Speech Emotion Recognition
Author: Generated Inference Script
Description: Complete inference pipeline for emotion prediction from audio files
"""

import os
import json
import logging
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import joblib
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_models_dir

class XGBSpeechEmotionInference:
    """XGBoost inference engine for speech emotion recognition"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize inference engine
        
        Args:
            model_path: Path to the model directory. If None, uses default path.
        """
        self.model_path = Path(model_path) if model_path else get_models_dir("speech_sentiment/xgb")
        self.setup_logging()
        self.load_models()
        self.load_config()
        
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs/inference")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'xgb_inference_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("XGBoost Speech Emotion Recognition Inference Engine Started")
        
    def load_models(self):
        """Load trained models and preprocessors"""
        self.logger.info(f"Loading models from {self.model_path}")
        
        try:
            # Load model components
            self.model = joblib.load(self.model_path / "model.pkl")
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            self.label_encoder = joblib.load(self.model_path / "label_encoder.pkl")
            
            self.logger.info("Models loaded successfully")
            self.logger.info(f"Model type: {type(self.model).__name__}")
            self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            self.logger.info(f"Classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
            
    def load_config(self):
        """Load training configuration"""
        try:
            config_path = self.model_path / "metrics" / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                self.logger.info("Configuration loaded successfully")
            else:
                # Default configuration
                self.config = {'n_mfcc': 13}
                self.logger.warning("Configuration file not found, using defaults")
        except Exception as e:
            self.logger.warning(f"Error loading configuration: {e}")
            self.config = {'n_mfcc': 13}
    
    def extract_features(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """Extract audio features from a single file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
            
            # Extract MFCC features
            mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config['n_mfcc']), axis=1)
            
            # Extract Chroma features
            chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            
            # Extract Spectral Contrast features
            contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            
            # Combine features
            features = {}
            
            # MFCC features
            for i in range(self.config['n_mfcc']):
                features[f'mfccs_mean_{i}'] = mfccs_mean[i]
            
            # Chroma features
            for i in range(12):
                features[f'chroma_mean_{i}'] = chroma_mean[i]
            
            # Contrast features
            for i in range(7):
                features[f'contrast_mean_{i}'] = contrast_mean[i]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features from {file_path}: {e}")
            raise
    
    def predict_single(self, file_path: Union[str, Path], 
                      return_probabilities: bool = False) -> Dict[str, Union[str, float, Dict]]:
        """Predict emotion for a single audio file
        
        Args:
            file_path: Path to the audio file
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        self.logger.info(f"Processing file: {file_path}")
        
        # Extract features
        features = self.extract_features(file_path)
        
        # Convert to DataFrame and scale
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        predicted_emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Create probability dictionary
        prob_dict = {
            emotion: float(prob) 
            for emotion, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        result = {
            'file_path': str(file_path),
            'predicted_emotion': predicted_emotion,
            'confidence': float(confidence),
            'prediction_label': int(prediction)
        }
        
        if return_probabilities:
            result['probabilities'] = prob_dict
            
        self.logger.info(f"Prediction: {predicted_emotion} (confidence: {confidence:.3f})")
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'model_path': str(self.model_path),
            'classes': list(self.label_encoder.classes_),
            'num_classes': len(self.label_encoder.classes_),
            'feature_count': self.model.n_features_in_,
            'config': self.config
        }
        
        # Try to get model parameters
        try:
            info['model_params'] = self.model.get_params()
        except:
            pass
            
        return info

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="XGBoost Speech Emotion Recognition Inference")
    parser.add_argument('--model-path', required=True, type=str, help='Path to model directory')
    parser.add_argument('--file',required=True ,type=str, help='Single audio file to process')
    parser.add_argument('--probabilities', action='store_true', help='Return class probabilities')
    parser.add_argument('--info', action='store_true', help='Show model information and exit')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        engine = XGBSpeechEmotionInference(args.model_path)
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        return
    
    # Show model info and exit
    if args.info:
        info = engine.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            if key != 'model_params':
                print(f"  {key}: {value}")
        return
    
    try:
        result = engine.predict_single(args.file, args.probabilities)
        print(f"\nPrediction for {args.file}:")
        print(f"Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if args.probabilities:
            print("Probabilities:")
            for emotion, prob in result['probabilities'].items():
                print(f"  {emotion}: {prob:.3f}")
    except Exception as e:
        print(f"Error processing file: {e}")
    return
    
if __name__ == "__main__":
    main()