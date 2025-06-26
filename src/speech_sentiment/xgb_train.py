#!/usr/bin/env python3
import json
import logging
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
)
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime
from utils import get_processed_data_dir, get_models_dir

class XGBSpeechEmotionTrainer:
    """XGBoost trainer for speech emotion recognition"""
    
    def __init__(self, config=None):
        """Initialize trainer with configuration"""
        self.config = config or self._get_default_config()
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.best_model = None
        
        # Metrics storage
        self.metrics = {}
        self.training_history = []
        
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'n_mfcc': 13,
            'param_grid': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 500, 1000],
            },
            'cv_folds': 5,
            'verbose': True
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'xgb_training_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("XGBoost Speech Emotion Recognition Training Started")
        
    def setup_directories(self):
        """Setup required directories"""
        self.model_dir = get_models_dir("speech_sentiment/xgb")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.model_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.model_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"Plots directory: {self.plots_dir}")
        self.logger.info(f"Metrics directory: {self.metrics_dir}")
    
    def load_data(self):
        """Load and prepare the dataset"""
        self.logger.info("Loading dataset...")
        
        try:
            train_csv = get_processed_data_dir("speech_sentiment") / "emotion_dataset.csv"
            self.df = pd.read_csv(train_csv)
            self.logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            
            # Display basic dataset info
            self.logger.info(f"Emotion distribution:\n{self.df['emotion'].value_counts()}")
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def encode_labels(self):
        """Encode emotion labels to numeric values"""
        self.logger.info("Encoding emotion labels...")
        
        self.df["emotion_label"] = self.label_encoder.fit_transform(self.df["emotion"])
        
        # Log label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        self.logger.info(f"Label mapping: {label_mapping}")
        
        # Save label mapping
        with open(self.metrics_dir / "label_mapping.json", 'w') as f:
            json.dump({k: str(v) for k,v in label_mapping.items()}, f, indent=2)
    
    def extract_features(self, file_path):
        """Extract audio features from a single file"""
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
            self.logger.warning(f"Error extracting features from {file_path}: {e}")
            # Return zero features in case of error
            features = {}
            for i in range(self.config['n_mfcc']):
                features[f'mfccs_mean_{i}'] = 0.0
            for i in range(12):
                features[f'chroma_mean_{i}'] = 0.0
            for i in range(7):
                features[f'contrast_mean_{i}'] = 0.0
            return features
    
    def extract_all_features(self):
        """Extract features from all audio files"""
        self.logger.info("Starting feature extraction...")
        
        # Check if features already exist
        features_path = get_processed_data_dir("speech_sentiment") / "xgb_features.csv"
        
        if features_path.exists():
            self.logger.info("Loading existing features...")
            self.df = pd.read_csv(features_path)
            self.logger.info(f"Features loaded. Shape: {self.df.shape}")
            return
        
        # Extract features with progress bar
        tqdm.pandas(desc="Extracting Features")
        features_list = self.df['path'].progress_apply(self.extract_features).tolist()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.logger.info(f"Features extracted. Shape: {features_df.shape}")
        
        # Combine with original dataframe
        self.df = pd.concat([self.df, features_df], axis=1)
        
        # Save features
        self.df.to_csv(features_path, index=False)
        self.logger.info(f"Features saved to {features_path}")
    
    def prepare_data(self):
        """Prepare training and testing data"""
        self.logger.info("Preparing training and testing data...")
        
        # Select feature columns (exclude non-feature columns)
        feature_columns = [col for col in self.df.columns 
                          if col not in ['path', 'emotion', 'emotion_label']]
        
        X = self.df[feature_columns]
        y = self.df['emotion_label']
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Target vector shape: {y.shape}")
        
        # Scale features
        self.logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y
        )
        
        self.logger.info(f"Training set shape: {self.X_train.shape}")
        self.logger.info(f"Testing set shape: {self.X_test.shape}")
        
        # Log class distribution
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        self.logger.info(f"Training set distribution: {train_dist.to_dict()}")
        self.logger.info(f"Testing set distribution: {test_dist.to_dict()}")
    
    def train_model(self):
        """Train XGBoost model with hyperparameter tuning"""
        self.logger.info("Starting model training with hyperparameter tuning...")
        
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(self.y_train)),
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        # Setup GridSearchCV
        self.logger.info("Performing grid search...")
        self.logger.info(f"Parameter grid: {self.config['param_grid']}")
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.config['param_grid'],
            cv=self.config['cv_folds'],
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 if self.config['verbose'] else 0
        )
        
        # Fit with progress tracking
        with tqdm(total=1, desc="Grid Search Progress") as pbar:
            grid_search.fit(self.X_train, self.y_train)
            pbar.update(1)
        
        self.best_model = grid_search.best_estimator_
        
        # Log best parameters
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Store training results
        self.metrics['best_params'] = grid_search.best_params_
        self.metrics['best_cv_score'] = grid_search.best_score_
        self.metrics['cv_results'] = grid_search.cv_results_
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        self.logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        # Store metrics
        self.metrics.update({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        })
        
        # Log results
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Test Precision: {precision:.4f}")
        self.logger.info(f"Test Recall: {recall:.4f}")
        self.logger.info(f"Test F1-Score: {f1:.4f}")
        
        # Detailed classification report
        class_report = classification_report(self.y_test, y_pred, 
                                           target_names=self.label_encoder.classes_)
        self.logger.info(f"Classification Report:\n{class_report}")
        
        return y_pred, y_pred_proba
    
    def create_visualizations(self, y_pred, y_pred_proba):
        """Create and save visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        self.logger.info("Creating confusion matrix...")
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_, 
                   cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        self.logger.info("Creating feature importance plot...")
        feature_importance = self.best_model.feature_importances_
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Get top 20 features
        top_indices = np.argsort(feature_importance)[-20:]
        top_importance = feature_importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_importance)), top_names)
        plt.xlabel('Feature Importance', fontsize=14)
        plt.title('Top 20 Feature Importances', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Class Distribution
        self.logger.info("Creating class distribution plot...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training distribution
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        train_labels = [self.label_encoder.classes_[i] for i in train_counts.index]
        ax1.bar(train_labels, train_counts.values)
        ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotion')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Testing distribution
        test_counts = pd.Series(self.y_test).value_counts().sort_index()
        test_labels = [self.label_encoder.classes_[i] for i in test_counts.index]
        ax2.bar(test_labels, test_counts.values)
        ax2.set_title('Testing Set Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Emotion')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Prediction Confidence Distribution
        self.logger.info("Creating prediction confidence plot...")
        max_proba = np.max(y_pred_proba, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_proba, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Maximum Prediction Probability', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Prediction Confidence Distribution', fontsize=16, fontweight='bold')
        plt.axvline(np.mean(max_proba), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_proba):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prediction_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("All visualizations created and saved")
    
    def save_models_and_metrics(self):
        """Save trained models and metrics"""
        self.logger.info("Saving models and metrics...")
        
        # Save models
        joblib.dump(self.best_model, self.model_dir / "model.pkl")
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        joblib.dump(self.label_encoder, self.model_dir / "label_encoder.pkl")
        
        # Save metrics
        with open(self.metrics_dir / "metrics.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_serializable[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    metrics_serializable[key] = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    metrics_serializable[key] = int(value)
                else:
                    metrics_serializable[key] = value
            
            json.dump(metrics_serializable, f, indent=2)
        
        # Save configuration
        with open(self.metrics_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Models saved to {self.model_dir}")
        self.logger.info(f"Metrics saved to {self.metrics_dir}")
    
    def generate_summary_report(self):
        """Generate a summary report"""
        self.logger.info("Generating summary report...")
        
        report = []
        report.append("=" * 60)
        report.append("XGBoost Speech Emotion Recognition - Training Summary")
        report.append("=" * 60)
        report.append(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("Dataset Information:")
        report.append(f"  - Total samples: {len(self.df)}")
        report.append(f"  - Number of classes: {len(self.label_encoder.classes_)}")
        report.append(f"  - Classes: {', '.join(self.label_encoder.classes_)}")
        report.append("")
        report.append("Model Performance:")
        report.append(f"  - Test Accuracy: {self.metrics['test_accuracy']:.4f}")
        report.append(f"  - Test Precision: {self.metrics['test_precision']:.4f}")
        report.append(f"  - Test Recall: {self.metrics['test_recall']:.4f}")
        report.append(f"  - Test F1-Score: {self.metrics['test_f1']:.4f}")
        report.append("")
        report.append("Best Hyperparameters:")
        for param, value in self.metrics['best_params'].items():
            report.append(f"  - {param}: {value}")
        report.append("")
        report.append("Files Generated:")
        report.append(f"  - Model: {self.model_dir / 'model.pkl'}")
        report.append(f"  - Scaler: {self.model_dir / 'scaler.pkl'}")
        report.append(f"  - Label Encoder: {self.model_dir / 'label_encoder.pkl'}")
        report.append(f"  - Metrics: {self.metrics_dir / 'metrics.json'}")
        report.append(f"  - Plots: {self.plots_dir}")
        report.append("=" * 60)
        
        summary_text = "\n".join(report)
        
        # Save report
        with open(self.model_dir / "training_summary.txt", 'w') as f:
            f.write(summary_text)
        
        # Log report
        self.logger.info("\n" + summary_text)
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            self.logger.info("Starting XGBoost training pipeline...")
            
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Encode labels
            self.encode_labels()
            
            # Step 3: Extract features
            self.extract_all_features()
            
            # Step 4: Prepare data
            self.prepare_data()
            
            # Step 5: Train model
            self.train_model()
            
            # Step 6: Evaluate model
            y_pred, y_pred_proba = self.evaluate_model()
            
            # Step 7: Create visualizations
            self.create_visualizations(y_pred, y_pred_proba)
            
            # Step 8: Save everything
            self.save_models_and_metrics()
            
            # Step 9: Generate summary
            self.generate_summary_report()
            
            self.logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training"""
    # Custom configuration (optional)
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'n_mfcc': 13,
        'param_grid': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 500, 1000],
        },
        'cv_folds': 5,
        'verbose': True
    }
    
    # Initialize and run trainer
    trainer = XGBSpeechEmotionTrainer(config)
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()