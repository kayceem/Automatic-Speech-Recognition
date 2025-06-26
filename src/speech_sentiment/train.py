#!/usr/bin/env python3
"""
Enhanced Speech Emotion Recognition Training Script
Features:
- Early stopping to prevent overfitting
- Memory optimization with batch processing
- On-the-fly data augmentation
- Learning rate scheduling
- Comprehensive logging and monitoring
"""

import os
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio.transforms as T

from speech_sentiment.feature import addNoise, getMELspectrogram, splitIntoChunks
from utils.helpers import get_models_dir
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
num_workers = multiprocessing.cpu_count() - 1

class SpeechEmotionDataset(Dataset):
    """Custom dataset for speech emotion recognition with on-the-fly processing"""
    
    def __init__(self, audio_paths: List[str], labels: List[int], sample_rate: int = 16000,
                 duration: float = 3.0, offset: float = 0.5, augment: bool = False,
                 scaler: Optional[StandardScaler] = None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset = offset
        self.augment = augment
        self.scaler = scaler
        # self.mel_transform = T.MelSpectrogram(
        #     sample_rate=self.sample_rate,
        #     n_fft=1024,
        #     win_length=512,
        #     hop_length=256,
        #     n_mels=128,
        #     f_max=self.sample_rate // 2,
        #     window_fn=torch.hamming_window
        # ).to('cuda')  # move transform to GPU

        self.db_transform = T.AmplitudeToDB(stype='power').to('cuda')
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        try:
            audio, _ = librosa.load(audio_path, duration=self.duration, 
                                 offset=self.offset, sr=self.sample_rate)
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            # Return silence if file cannot be loaded
            audio = np.zeros(int(self.sample_rate * self.duration))
        
        # Pad or truncate to fixed length
        signal = np.zeros(int(self.sample_rate * self.duration))
        signal[:len(audio)] = audio
        
        # Apply augmentation if enabled
        if self.augment:
            signal = self._add_noise(signal)
        
        # Convert to mel spectrogram
        mel_spec = self._get_mel_spectrogram(signal)
        
        # Split into chunks
        chunks = self._split_into_chunks(mel_spec, win_size=128, stride=64)
        
        # Add channel dimension
        chunks = np.expand_dims(chunks, 1)
        
        # Apply scaling if scaler is provided
        if self.scaler is not None:
            t, c, h, w = chunks.shape
            chunks_flat = chunks.reshape(1, -1)
            chunks_flat = self.scaler.transform(chunks_flat)
            chunks = chunks_flat.reshape(t, c, h, w)
        
        return torch.FloatTensor(chunks), torch.LongTensor([label])
    
    def _add_noise(self, signal: np.ndarray, snr_low: int = 15, snr_high: int = 30) -> np.ndarray:
        return addNoise(signal, snr_low, snr_high)
    
    def _get_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        return getMELspectrogram(audio, self.sample_rate)

    def _split_into_chunks(self, mel_spec: np.ndarray, win_size: int, stride: int) -> np.ndarray:
        return splitIntoChunks(mel_spec, win_size, stride)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class LRScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(self, optimizer, mode='reduce_on_plateau', factor=0.5, patience=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_score = None
        self.counter = 0
        
    def step(self, score):
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self._reduce_lr()
            self.counter = 0
            
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if new_lr < old_lr:
                logger.info(f'Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')

def load_and_prepare_data(csv_path: str, train_ratio: float = 0.8, 
                         val_ratio: float = 0.1) -> Tuple[Dict, Dict]:
    """Load and split data into train/val/test sets"""
    
    logger.info(f"Loading data from {csv_path}")
    data = pd.read_csv(csv_path)
    
    # Create emotion mapping
    unique_emotions = data.emotion.unique()
    emotions = {emotion: i for i, emotion in enumerate(unique_emotions)}
    
    # Clean and map emotions
    data['emotion'] = data['emotion'].str.strip().str.lower()
    data['emotion'] = data['emotion'].map(emotions)
    
    logger.info(f"Found {len(emotions)} emotions: {list(emotions.keys())}")
    logger.info(f"Total samples: {len(data)}")
    
    # Split data by emotion to maintain class balance
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for emotion_idx in range(len(emotions)):
        emotion_data = data[data.emotion == emotion_idx]
        indices = np.random.permutation(len(emotion_data))
        
        n_train = int(train_ratio * len(indices))
        n_val = int(val_ratio * len(indices))
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_paths.extend(emotion_data.iloc[train_idx]['path'].tolist())
        train_labels.extend([emotion_idx] * len(train_idx))
        
        val_paths.extend(emotion_data.iloc[val_idx]['path'].tolist())
        val_labels.extend([emotion_idx] * len(val_idx))
        
        test_paths.extend(emotion_data.iloc[test_idx]['path'].tolist())
        test_labels.extend([emotion_idx] * len(test_idx))
    
    data_splits = {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }
    
    logger.info(f"Data splits - Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
    
    return data_splits, emotions
def fit_scaler_full(train_paths: List[str], train_labels: List[int]) -> StandardScaler:
    """Fit scaler on the full training data (flattened mel spectrogram chunks)"""
    logger.info("Fitting scaler on full training data...")

    dataset = SpeechEmotionDataset(train_paths, train_labels, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    all_features = []

    for batch_idx, (features, _) in enumerate(tqdm(loader, desc="Extracting features for scaler")):
        # Flatten features
        features_flat = features.reshape(1, -1)
        all_features.append(features_flat.numpy())

    all_features = np.vstack(all_features)
    scaler = StandardScaler()
    scaler.fit(all_features)

    logger.info("Scaler fitting completed on full dataset.")
    return scaler

def fit_scaler(train_paths: List[str], train_labels: List[int], 
               sample_size: int = 1000) -> StandardScaler:
    """Fit scaler on a subset of training data to save memory"""
    
    logger.info("Fitting scaler on training data subset...")
    
    # Use a subset for fitting scaler to save memory
    indices = np.random.choice(len(train_paths), min(sample_size, len(train_paths)), replace=False)
    subset_paths = [train_paths[i] for i in indices]
    subset_labels = [train_labels[i] for i in indices]
    
    # Create temporary dataset without augmentation
    temp_dataset = SpeechEmotionDataset(subset_paths, subset_labels, augment=False)
    temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
    
    # Collect features for fitting
    all_features = []
    for batch_idx, (features, _) in enumerate(temp_loader):
        if batch_idx >= sample_size:
            break
        # Flatten features
        features_flat = features.reshape(1, -1)
        all_features.append(features_flat.numpy())
        
        if batch_idx % 100 == 0:
            logger.info(f"Processed {batch_idx}/{min(sample_size, len(subset_paths))} samples for scaler fitting")
    
    all_features = np.vstack(all_features)
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    logger.info("Scaler fitting completed")
    return scaler

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    features, labels = zip(*batch)
    
    # Find max sequence length in batch
    max_seq_len = max(f.shape[0] for f in features)
    
    # Pad sequences to max length
    padded_features = []
    for f in features:
        if f.shape[0] < max_seq_len:
            padding = torch.zeros(max_seq_len - f.shape[0], *f.shape[1:])
            f = torch.cat([f, padding], dim=0)
        padded_features.append(f)
    
    features = torch.stack(padded_features)
    labels = torch.cat(labels)
    
    return features, labels

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output_logits, output_softmax, _ = model(data)
        loss = criterion(output_logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output_softmax.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output_logits, output_softmax, _ = model(data)
            loss = criterion(output_logits, target)
            
            total_loss += loss.item()
            pred = output_softmax.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_preds, all_targets


def plot_confusion_matrix(y_true, y_pred, emotions, save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    emotion_names = list(emotions.keys())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to emotion dataset CSV')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create model directory
    model_dir = Path(args.model_dir) if args.model_dir else get_models_dir("speech_sentiment/cnn_bilstm")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    data_splits, emotions = load_and_prepare_data(args.data_path)
    
    # Load or fit scaler
    scaler_path = model_dir / 'scaler.pkl'
    train_paths, train_labels = data_splits['train']
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded existing scaler from {scaler_path}")
    else:
        logger.info("Fitting new scaler...")
        # scaler = fit_scaler(train_paths, train_labels)
        scaler = fit_scaler_full(train_paths, train_labels)
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler fitted and saved to {scaler_path}")

    # Create datasets
    train_dataset = SpeechEmotionDataset(train_paths, train_labels, augment=True, scaler=scaler)
    val_dataset = SpeechEmotionDataset(*data_splits['val'], augment=False, scaler=scaler)
    test_dataset = SpeechEmotionDataset(*data_splits['test'], augment=False, scaler=scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=custom_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=custom_collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=custom_collate_fn, num_workers=num_workers)
    
    # Import model (assuming it's available)
    try:
        from speech_sentiment.model import HybridModel, loss_fnc
    except ImportError:
        logger.error("Could not import HybridModel. Please ensure the model module is available.")
        return
    
    # Initialize model
    model = HybridModel(num_emotions=len(emotions)).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup training components
    criterion = loss_fnc
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
    
    # Initialize schedulers and early stopping
    lr_scheduler = LRScheduler(optimizer, patience=5)
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        lr_scheduler.step(val_loss)
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = model_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'emotions': emotions
            }, model_path)
            logger.info(f"Best model saved to {model_path}")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Load best model for testing
    checkpoint = torch.load(model_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_targets = validate_epoch(model, test_loader, criterion, device)
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Generate classification report
    emotion_names = list(emotions.keys())
    report = classification_report(test_targets, test_preds, target_names=emotion_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plot_confusion_matrix(test_targets, test_preds, emotions, 
                         save_path=model_dir / 'confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(model_dir / 'training_history.png')
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()