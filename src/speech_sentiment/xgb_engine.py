#!/usr/bin/env python3
"""
XGBoost Inference Engine with Calibrated Uncertainties
Supports Platt scaling and isotonic regression for uncertainty estimation
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
    """XGBoost inference engine with calibrated uncertainty estimates"""
    
    def __init__(self, model_path: Optional[str] = None, use_calibration: bool = True):
        """Initialize inference engine
        
        Args:
            model_path: Path to the model directory. If None, uses default path.
            use_calibration: Whether to use calibrated model for uncertainty estimates
        """
        self.model_path = Path(model_path) if model_path else get_models_dir("speech_sentiment/xgb")
        self.use_calibration = use_calibration
        self.setup_logging()
        self.load_models()
        self.load_config()
        self.load_operating_points()
        
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
            # Load base model
            self.model = joblib.load(self.model_path / "model.pkl")
            
            # Load calibration data if available
            calibration_path = self.model_path / "calibration_data.pkl"
            if self.use_calibration and calibration_path.exists():
                calibration_data = joblib.load(calibration_path)
                self.calibration_models = calibration_data['calibration_models']
                self.calibration_method = calibration_data['calibration_method']
                self.logger.info(f"Calibration loaded ({self.calibration_method})")
                self.is_calibrated = True
            else:
                self.calibration_models = None
                self.calibration_method = None
                self.is_calibrated = False
                if self.use_calibration:
                    self.logger.warning("Calibration data not found, using base model")
            
            # Load preprocessors
            self.scaler = joblib.load(self.model_path / "scaler.pkl")
            self.label_encoder = joblib.load(self.model_path / "label_encoder.pkl")
            
            self.logger.info("Models loaded successfully")
            self.logger.info(f"Model type: {type(self.model).__name__}")
            self.logger.info(f"Calibration: {'Enabled' if self.is_calibrated else 'Disabled'}")
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
                self.config = {'n_mfcc': 13}
                self.logger.warning("Configuration file not found, using defaults")
        except Exception as e:
            self.logger.warning(f"Error loading configuration: {e}")
            self.config = {'n_mfcc': 13}
    
    def load_operating_points(self):
        """Load operating thresholds for decision making"""
        try:
            op_path = self.model_path / "operating_points.json"
            if op_path.exists():
                with open(op_path, 'r') as f:
                    self.operating_points = json.load(f)
                self.logger.info("Operating points loaded successfully")
            else:
                # Default operating points based on confidence thresholds
                self.operating_points = {
                    'high_confidence': 0.80,  # Direct prediction
                    'medium_confidence': 0.60,  # Flag for review
                    'low_confidence': 0.40,  # Reject/defer
                    'per_class_thresholds': {}  # Can be class-specific
                }
                self.logger.warning("Operating points not found, using defaults")
        except Exception as e:
            self.logger.warning(f"Error loading operating points: {e}")
            self.operating_points = {
                'high_confidence': 0.80,
                'medium_confidence': 0.60,
                'low_confidence': 0.40,
                'per_class_thresholds': {}
            }
    
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
    
    def compute_uncertainty_metrics(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Compute various uncertainty metrics from prediction probabilities
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Dictionary of uncertainty metrics
        """
        # Predictive entropy (uncertainty)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Margin (difference between top two predictions)
        sorted_probs = np.sort(probabilities)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        
        # Confidence (max probability)
        confidence = np.max(probabilities)
        
        # Normalized entropy
        max_entropy = np.log(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'confidence': float(confidence),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'margin': float(margin),
            'uncertainty': 1.0 - float(confidence)  # Simple uncertainty measure
        }
    
    def get_decision_recommendation(self, uncertainty_metrics: Dict[str, float], 
                                   predicted_emotion: str) -> Dict[str, str]:
        """Get decision recommendation based on uncertainty and operating points
        
        Args:
            uncertainty_metrics: Dictionary of uncertainty metrics
            predicted_emotion: Predicted emotion class
            
        Returns:
            Dictionary with decision recommendation
        """
        confidence = uncertainty_metrics['confidence']
        
        # Check for class-specific threshold
        class_threshold = self.operating_points['per_class_thresholds'].get(
            predicted_emotion, None
        )
        
        # Determine decision based on confidence thresholds
        if class_threshold and confidence >= class_threshold:
            decision = 'accept'
            reason = f'Class-specific threshold met ({class_threshold:.2f})'
        elif confidence >= self.operating_points['high_confidence']:
            decision = 'accept'
            reason = 'High confidence prediction'
        elif confidence >= self.operating_points['medium_confidence']:
            decision = 'review'
            reason = 'Medium confidence - recommend human review'
        elif confidence >= self.operating_points['low_confidence']:
            decision = 'uncertain'
            reason = 'Low confidence - uncertain prediction'
        else:
            decision = 'reject'
            reason = 'Very low confidence - reject prediction'
        
        return {
            'decision': decision,
            'reason': reason,
            'confidence_threshold_used': class_threshold or self.operating_points['high_confidence']
        }
    
    def predict_single(self, file_path: Union[str, Path], 
                      return_probabilities: bool = False,
                      return_uncertainty: bool = True) -> Dict[str, Union[str, float, Dict]]:
        """Predict emotion for a single audio file with calibrated uncertainties
        
        Args:
            file_path: Path to the audio file
            return_probabilities: Whether to return class probabilities
            return_uncertainty: Whether to return uncertainty metrics
            
        Returns:
            Dictionary containing prediction results with uncertainties
        """
        self.logger.info(f"Processing file: {file_path}")
        
        # Extract features
        features = self.extract_features(file_path)
        
        # Convert to DataFrame and scale
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction using active model (calibrated or base)
        prediction = self.active_model.predict(features_scaled)[0]
        predicted_emotion = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get calibrated prediction probabilities
        probabilities = self.active_model.predict_proba(features_scaled)[0]
        
        # Compute uncertainty metrics
        uncertainty_metrics = self.compute_uncertainty_metrics(probabilities)
        
        # Get decision recommendation based on operating points
        decision_info = self.get_decision_recommendation(
            uncertainty_metrics, predicted_emotion
        )
        
        # Create probability dictionary
        prob_dict = {
            emotion: float(prob) 
            for emotion, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        result = {
            'file_path': str(file_path),
            'predicted_emotion': predicted_emotion,
            'prediction_label': int(prediction),
            'calibration_used': self.calibrated_model is not None
        }
        
        if return_uncertainty:
            result['uncertainty_metrics'] = uncertainty_metrics
            result['decision'] = decision_info
        
        if return_probabilities:
            result['probabilities'] = prob_dict
            
        self.logger.info(
            f"Prediction: {predicted_emotion} "
            f"(confidence: {uncertainty_metrics['confidence']:.3f}, "
            f"decision: {decision_info['decision']})"
        )
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'model_path': str(self.model_path),
            'calibration_enabled': self.calibrated_model is not None,
            'calibration_method': 'Platt scaling' if self.calibrated_model else 'None',
            'classes': list(self.label_encoder.classes_),
            'num_classes': len(self.label_encoder.classes_),
            'feature_count': self.model.n_features_in_,
            'config': self.config,
            'operating_points': self.operating_points
        }
        
        return info

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="XGBoost Speech Emotion Recognition Inference with Calibrated Uncertainties"
    )
    parser.add_argument('--model-path', required=True, type=str, help='Path to model directory')
    parser.add_argument('--file', required=True, type=str, help='Single audio file to process')
    parser.add_argument('--probabilities', action='store_true', help='Return class probabilities')
    parser.add_argument('--no-calibration', action='store_true', help='Disable calibration')
    parser.add_argument('--info', action='store_true', help='Show model information and exit')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        engine = XGBSpeechEmotionInference(
            args.model_path, 
            use_calibration=not args.no_calibration
        )
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
        print(f"Calibration: {'Enabled' if result['calibration_used'] else 'Disabled'}")
        
        if 'uncertainty_metrics' in result:
            print("\nUncertainty Metrics:")
            for metric, value in result['uncertainty_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        if 'decision' in result:
            print(f"\nDecision: {result['decision']['decision'].upper()}")
            print(f"Reason: {result['decision']['reason']}")
        
        if args.probabilities and 'probabilities' in result:
            print("\nClass Probabilities:")
            for emotion, prob in sorted(
                result['probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                print(f"  {emotion}: {prob:.4f}")
                
    except Exception as e:
        print(f"Error processing file: {e}")
    
if __name__ == "__main__":
    main()