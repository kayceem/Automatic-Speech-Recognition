#!/usr/bin/env python3
"""
Calibration Script for Existing XGBoost Model
This script calibrates an already-trained model without retraining
"""

import json
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from utils import get_processed_data_dir, get_models_dir

class ModelCalibrator:
    """Calibrate existing trained model"""
    
    def __init__(self, model_dir: str):
        """
        Initialize calibrator
        
        Args:
            model_dir: Path to directory containing trained model
        """
        self.model_dir = Path(model_dir)
        self.setup_logging()
        self.load_existing_model()
        
    def setup_logging(self):
        """Setup logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs/calibration")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'calibration_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model Calibration Started")
    
    def load_existing_model(self):
        """Load existing trained model and preprocessors"""
        self.logger.info(f"Loading existing model from {self.model_dir}")
        
        try:
            # Load model components
            self.model = joblib.load(self.model_dir / "model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")
            
            self.logger.info("✓ Existing model loaded successfully")
            self.logger.info(f"  Model type: {type(self.model).__name__}")
            self.logger.info(f"  Classes: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def load_and_prepare_data(self, test_size=0.3, random_state=42):
        """
        Load data and create calibration + test sets
        
        Args:
            test_size: Proportion to use for test (remaining goes to calibration)
            random_state: Random seed
        """
        self.logger.info("Loading data for calibration...")
        
        try:
            # Load features (assuming they exist from training)
            features_path = get_processed_data_dir("speech_sentiment") / "xgb_features.csv"
            df = pd.read_csv(features_path)
            
            self.logger.info(f"Data loaded. Shape: {df.shape}")
            
            # Select feature columns
            feature_columns = [col for col in df.columns 
                              if col not in ['path', 'emotion', 'emotion_label']]
            
            X = df[feature_columns]
            y = df['emotion_label']
            
            # Scale features (using existing scaler)
            X_scaled = self.scaler.transform(X)
            
            # Split into calibration and test sets
            # We use test_size for final test, rest for calibration
            self.X_cal, self.X_test, self.y_cal, self.y_test = train_test_split(
                X_scaled, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            self.logger.info(f"Calibration set shape: {self.X_cal.shape}")
            self.logger.info(f"Test set shape: {self.X_test.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def calibrate_model(self, method='sigmoid'):
        """
        Calibrate the model using specified method
        
        Args:
            method: 'sigmoid' (Platt scaling) or 'isotonic'
        """
        self.logger.info(f"Calibrating model using {method} method...")
        
        try:
            # Fix: Check if model is properly configured as classifier
            # If not, we need to manually set up calibration
            
            # Get uncalibrated probabilities on calibration set
            y_proba_uncal = self.model.predict_proba(self.X_cal)
            
            # Manual Platt scaling implementation (more robust)
            if method == 'sigmoid':
                from sklearn.linear_model import LogisticRegression
                
                # For multi-class, we use one-vs-rest approach
                n_classes = len(self.label_encoder.classes_)
                self.calibration_models = []
                
                for class_idx in range(n_classes):
                    # Binary labels for this class
                    y_binary = (self.y_cal == class_idx).astype(int)
                    
                    # Uncalibrated probabilities for this class (reshape for sklearn)
                    proba_class = y_proba_uncal[:, class_idx].reshape(-1, 1)
                    
                    # Fit logistic regression for calibration
                    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
                    lr.fit(proba_class, y_binary)
                    
                    self.calibration_models.append(lr)
                
                self.calibration_method = 'sigmoid_manual'
                self.logger.info("✓ Manual Platt scaling calibration complete")
                
            elif method == 'isotonic':
                from sklearn.isotonic import IsotonicRegression
                
                n_classes = len(self.label_encoder.classes_)
                self.calibration_models = []
                
                for class_idx in range(n_classes):
                    y_binary = (self.y_cal == class_idx).astype(int)
                    proba_class = y_proba_uncal[:, class_idx]
                    
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(proba_class, y_binary)
                    
                    self.calibration_models.append(iso)
                
                self.calibration_method = 'isotonic_manual'
                self.logger.info("✓ Manual isotonic regression calibration complete")
            
            # Evaluate calibration improvement
            self.evaluate_calibration()
            
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            raise
    
    def predict_calibrated(self, X):
        """
        Make calibrated predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Calibrated probabilities
        """
        # Get base model probabilities
        y_proba_uncal = self.model.predict_proba(X)
        
        # Apply calibration
        n_classes = len(self.label_encoder.classes_)
        y_proba_cal = np.zeros_like(y_proba_uncal)
        
        for class_idx in range(n_classes):
            proba_class = y_proba_uncal[:, class_idx]
            
            if self.calibration_method == 'sigmoid_manual':
                # Reshape for sklearn
                proba_class_reshaped = proba_class.reshape(-1, 1)
                y_proba_cal[:, class_idx] = self.calibration_models[class_idx].predict_proba(
                    proba_class_reshaped
                )[:, 1]
            elif self.calibration_method == 'isotonic_manual':
                y_proba_cal[:, class_idx] = self.calibration_models[class_idx].predict(proba_class)
        
        # Normalize to sum to 1
        y_proba_cal = y_proba_cal / y_proba_cal.sum(axis=1, keepdims=True)
        
        return y_proba_cal
    
    def evaluate_calibration(self):
        """Evaluate calibration improvement"""
        self.logger.info("Evaluating calibration improvement...")
        
        # Convert y_test to numpy array to avoid pandas indexing issues
        y_test_array = np.array(self.y_test)
        
        # Get predictions on test set
        y_pred_base = self.model.predict(self.X_test)
        y_proba_base = self.model.predict_proba(self.X_test)
        
        # Get calibrated predictions using our custom method
        y_proba_cal = self.predict_calibrated(self.X_test)
        y_pred_cal = np.argmax(y_proba_cal, axis=1)
        
        # Calculate metrics
        acc_base = accuracy_score(y_test_array, y_pred_base)
        acc_cal = accuracy_score(y_test_array, y_pred_cal)
        
        logloss_base = log_loss(y_test_array, y_proba_base)
        logloss_cal = log_loss(y_test_array, y_proba_cal)
        
        # Store results
        self.calibration_results = {
            'accuracy_base': float(acc_base),
            'accuracy_calibrated': float(acc_cal),
            'log_loss_base': float(logloss_base),
            'log_loss_calibrated': float(logloss_cal),
            'log_loss_improvement': float(logloss_base - logloss_cal),
            'improvement_percentage': float((logloss_base - logloss_cal) / logloss_base * 100)
        }
        
        # Log results
        self.logger.info("Calibration Results:")
        self.logger.info(f"  Base Model Accuracy: {acc_base:.4f}")
        self.logger.info(f"  Calibrated Model Accuracy: {acc_cal:.4f}")
        self.logger.info(f"  Base Model Log Loss: {logloss_base:.4f}")
        self.logger.info(f"  Calibrated Model Log Loss: {logloss_cal:.4f}")
        self.logger.info(f"  Improvement: {logloss_base - logloss_cal:.4f} ({self.calibration_results['improvement_percentage']:.2f}%)")

    def compute_operating_points(self):
        """Compute operating points based on calibrated uncertainties"""
        self.logger.info("Computing operating points...")
        
        # Get calibrated predictions using our custom method
        y_proba_cal = self.predict_calibrated(self.X_test)
        y_pred_cal = np.argmax(y_proba_cal, axis=1)
        
        # Convert y_test to numpy array to avoid pandas indexing issues
        y_test_array = np.array(self.y_test)
        
        # Calculate confidence scores
        confidences = np.max(y_proba_cal, axis=1)
        correct = (y_pred_cal == y_test_array).astype(int)
        
        # Analyze confidence-accuracy relationship
        confidence_bins = np.arange(0, 1.1, 0.1)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
                bin_counts.append(int(mask.sum()))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        # Find optimal thresholds
        sorted_indices = np.argsort(confidences)[::-1]
        cumulative_accuracy = np.cumsum(correct[sorted_indices]) / np.arange(1, len(correct) + 1)
        
        # Default thresholds
        high_conf_threshold = 0.80
        medium_conf_threshold = 0.60
        low_conf_threshold = 0.40
        
        # Find where accuracy drops below 90%
        high_idx = np.where(cumulative_accuracy < 0.90)[0]
        if len(high_idx) > 0:
            high_conf_threshold = float(confidences[sorted_indices[high_idx[0]]])
        
        # Per-class thresholds
        per_class_thresholds = {}
        for class_idx in range(len(self.label_encoder.classes_)):
            class_mask = y_test_array == class_idx
            if class_mask.sum() > 10:  # Only if sufficient samples
                class_confidences = confidences[class_mask]
                class_correct = correct[class_mask]
                
                sorted_class_idx = np.argsort(class_confidences)[::-1]
                cum_acc = np.cumsum(class_correct[sorted_class_idx]) / np.arange(1, len(class_correct) + 1)
                threshold_idx = np.where(cum_acc < 0.80)[0]
                if len(threshold_idx) > 0:
                    per_class_thresholds[self.label_encoder.classes_[class_idx]] = float(
                        class_confidences[sorted_class_idx[threshold_idx[0]]]
                    )
        
        self.operating_points = {
            'high_confidence': high_conf_threshold,
            'medium_confidence': medium_conf_threshold,
            'low_confidence': low_conf_threshold,
            'per_class_thresholds': per_class_thresholds,
            'confidence_accuracy_relationship': {
                'bins': confidence_bins.tolist(),
                'accuracies': bin_accuracies,
                'counts': bin_counts
            }
        }
        
        self.logger.info(f"Operating points computed:")
        self.logger.info(f"  High confidence: {high_conf_threshold:.3f}")
        self.logger.info(f"  Medium confidence: {medium_conf_threshold:.3f}")
        self.logger.info(f"  Low confidence: {low_conf_threshold:.3f}")

    def create_calibration_plots(self):
        """Create calibration visualization plots"""
        self.logger.info("Creating calibration plots...")
        
        plots_dir = self.model_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Get probabilities
        y_proba_base = self.model.predict_proba(self.X_test)
        y_proba_cal = self.predict_calibrated(self.X_test)
        
        # Convert y_test to numpy array to avoid pandas indexing issues
        y_test_array = np.array(self.y_test)
        
        n_classes = len(self.label_encoder.classes_)
        
        # 1. Calibration curves for each class
        fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, class_name in enumerate(self.label_encoder.classes_):
            y_binary = (y_test_array == idx).astype(int)
            
            prob_base = y_proba_base[:, idx]
            prob_cal = y_proba_cal[:, idx]
            
            # Compute calibration curves
            fraction_pos_base, mean_pred_base = calibration_curve(
                y_binary, prob_base, n_bins=10, strategy='uniform'
            )
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_binary, prob_cal, n_bins=10, strategy='uniform'
            )
            
            # Plot
            axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.5)
            axes[idx].plot(mean_pred_base, fraction_pos_base, 's-', 
                        label=f'Base (Brier: {brier_score_loss(y_binary, prob_base):.3f})')
            axes[idx].plot(mean_pred_cal, fraction_pos_cal, 'o-',
                        label=f'Calibrated (Brier: {brier_score_loss(y_binary, prob_cal):.3f})')
            axes[idx].set_xlabel('Mean Predicted Probability')
            axes[idx].set_ylabel('Fraction of Positives')
            axes[idx].set_title(f'{class_name}')
            axes[idx].legend(loc='lower right', fontsize=8)
            axes[idx].grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Calibration Curves: Before vs After', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        max_proba_base = np.max(y_proba_base, axis=1)
        max_proba_cal = np.max(y_proba_cal, axis=1)
        
        ax1.hist(max_proba_base, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(max_proba_base), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_proba_base):.3f}')
        ax1.set_xlabel('Maximum Prediction Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Base Model Confidence')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.hist(max_proba_cal, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax2.axvline(np.mean(max_proba_cal), color='red', linestyle='--',
                label=f'Mean: {np.mean(max_proba_cal):.3f}')
        ax2.set_xlabel('Maximum Prediction Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Calibrated Model Confidence')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Plots saved to {plots_dir}")

    def save_results(self):
        """Save calibrated model and results"""
        self.logger.info("Saving calibrated model and results...")
        
        # Save calibration models (the per-class calibrators)
        joblib.dump({
            'calibration_models': self.calibration_models,
            'calibration_method': self.calibration_method,
            'base_model': self.model
        }, self.model_dir / "calibrated_model.pkl")
        
        # Save operating points
        with open(self.model_dir / "operating_points.json", 'w') as f:
            json.dump(self.operating_points, f, indent=2)
        
        # Save calibration results
        metrics_dir = self.model_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        with open(metrics_dir / "calibration_results.json", 'w') as f:
            json.dump(self.calibration_results, f, indent=2)
        
        self.logger.info("✓ All results saved:")
        self.logger.info(f"  - Calibrated model: {self.model_dir / 'calibrated_model.pkl'}")
        self.logger.info(f"  - Operating points: {self.model_dir / 'operating_points.json'}")
        self.logger.info(f"  - Calibration metrics: {metrics_dir / 'calibration_results.json'}")
        
    def run_calibration_pipeline(self, method='sigmoid', test_size=0.3):
        """
        Run complete calibration pipeline
        
        Args:
            method: 'sigmoid' (Platt scaling) or 'isotonic'
            test_size: Proportion of data for final test
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting Calibration Pipeline")
            self.logger.info("=" * 60)
            
            # Step 1: Load and prepare data
            self.load_and_prepare_data(test_size=test_size)
            
            # Step 2: Calibrate model
            self.calibrate_model(method=method)
            
            # Step 3: Compute operating points
            self.compute_operating_points()
            
            # Step 4: Create visualizations
            self.create_calibration_plots()
            
            # Step 5: Save results
            self.save_results()
            
            self.logger.info("=" * 60)
            self.logger.info("Calibration Pipeline Completed Successfully!")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Calibration pipeline failed: {e}")
            raise


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate existing XGBoost model")
    parser.add_argument('--model-dir', required=True, type=str,
                       help='Path to directory containing trained model')
    parser.add_argument('--method', default='sigmoid', choices=['sigmoid', 'isotonic'],
                       help='Calibration method (sigmoid=Platt scaling)')
    parser.add_argument('--test-size', default=0.3, type=float,
                       help='Proportion of data for test set')
    
    args = parser.parse_args()
    
    # Run calibration
    calibrator = ModelCalibrator(args.model_dir)
    calibrator.run_calibration_pipeline(method=args.method, test_size=args.test_size)
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print(f"Log Loss Improvement: {calibrator.calibration_results['log_loss_improvement']:.4f}")
    print(f"Percentage Improvement: {calibrator.calibration_results['improvement_percentage']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()