import pytorch_lightning as pl
import json
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from torch.utils.data import DataLoader, Dataset

from asr.helpers import TextTransform

class MelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, hop_length=160, device='cpu'):
        super(MelSpec, self).__init__()
        self.device = device
        self.transform = transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length
        ).to(self.device)

    def forward(self, x):
        return self.transform(x.to(self.device))


# For Engine Inference Only
def get_featurizer(sample_rate=16000, n_mels=80, hop_length=160, device='cpu'):
    return MelSpec(sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length, device=device)


# Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, json_path, log_ex=True, valid=False, augmentation_config=None):
        print(f"Loading json data from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.text_process = TextTransform()
        self.log_ex = log_ex

        # Default augmentation config
        if augmentation_config is None:
            augmentation_config = {
                'use_spec_augment': True,
                'use_time_mask': True,
                'use_freq_mask': True,
                'use_time_stretch': False,
                'use_noise_mixing': False,
                'time_mask_param': 15,
                'time_mask_prob': 0.05,
                'time_mask_count': 10,
                'freq_mask_param': 27,
                'time_stretch_rate': 1.0,
                'noise_snr_db': 20.0
            }
        
        self.augmentation_config = augmentation_config

        if valid:
            # No augmentation for validation
            self.audio_transforms = torch.nn.Sequential(MelSpec())
        else:
            # Build augmentation pipeline based on config
            self.audio_transforms = self._build_augmentation_pipeline()

    def _build_augmentation_pipeline(self):
        """Build augmentation pipeline based on config flags"""
        transforms_list = [MelSpec()]
        if self.augmentation_config.get('use_time_stretch', False):
            stretch_rate = self.augmentation_config.get('time_stretch_rate', 1.0)
            if stretch_rate != 1.0:
                transforms_list.append(
                    transforms.TimeStretch(
                        n_freq=self.augmentation_config.get('n_mels', 80),
                        fixed_rate=stretch_rate
                    )
                )
        
        if self.augmentation_config.get('use_spec_augment', True):
            # Frequency Masking
            if self.augmentation_config.get('use_freq_mask', True):
                transforms_list.append(
                    transforms.FrequencyMasking(
                        freq_mask_param=self.augmentation_config.get('freq_mask_param', 27)
                    )
                )
            
            # Time Masking
            if self.augmentation_config.get('use_time_mask', True):
                time_mask_count = self.augmentation_config.get('time_mask_count', 10)
                time_mask_param = self.augmentation_config.get('time_mask_param', 15)
                time_mask_prob = self.augmentation_config.get('time_mask_prob', 0.05)
                
                time_masks = [
                    torchaudio.transforms.TimeMasking(
                        time_mask_param=time_mask_param, 
                        p=time_mask_prob
                    ) for _ in range(time_mask_count)
                ]
                transforms_list.extend(time_masks)
        
        return nn.Sequential(*transforms_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        file_path = item["key"]

        try:
            waveform, sample_rate = torchaudio.load(file_path, backend='soundfile')
            
            # Apply noise mixing if enabled (before mel spectrogram)
            if self.augmentation_config.get('use_noise_mixing', False):
                snr_db = self.augmentation_config.get('noise_snr_db', 20.0)
                noise = torch.randn_like(waveform) * 0.005
                signal_power = waveform.pow(2).mean()
                noise_power = noise.pow(2).mean()
                snr = 10 ** (snr_db / 10)
                noise = noise * torch.sqrt(signal_power / (snr * noise_power))
                waveform = waveform + noise
            
            utterance = item["text"].lower()
            label = self.text_process.text_to_int(utterance)
            spectrogram = self.audio_transforms(waveform)
            label_len = len(label)

            if spectrogram.shape[0] > 1:
                raise Exception("dual channel, skipping audio file %s" % file_path)
            if spectrogram.shape[2] > 4096 * 2:
                raise Exception("spectrogram too big. size %s" % spectrogram.shape[2])
            if label_len == 0:
                raise Exception("label len is zero... skipping %s" % file_path)

            return spectrogram, label, label_len

        except Exception as e:
            if self.log_ex:
                print(f"{str(e)}\r", end="")
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)

    def describe(self):
        return self.data.describe()


# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_json, test_json, num_workers, augmentation_config=None):
        super().__init__()
        self.batch_size = batch_size
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers
        self.augmentation_config = augmentation_config

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(
            self.train_json, 
            valid=False, 
            augmentation_config=self.augmentation_config
        )
        self.test_dataset = CustomAudioDataset(self.test_json, valid=True)

    def data_processing(self, data):
        spectrograms, labels, input_lengths, label_lengths = [],[],[],[]
        for spectrogram, label, label_length in data:
            if spectrogram is None:
                continue
            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(torch.Tensor(label))
            input_lengths.append(((spectrogram.shape[-1] - 1) // 2 - 1) // 2)
            label_lengths.append(label_length)

        # Pad the spectrograms to have the same width (time dimension)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        # Convert input_lengths and label_lengths to tensors
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        mask = torch.ones(
            spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1]
        )
        for i, l in enumerate(input_lengths):
            mask[i, :, :l] = 0

        return spectrograms, labels, input_lengths, label_lengths, mask.bool()
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: self.data_processing(x),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.data_processing(x),
            num_workers=self.num_workers,
            pin_memory=True,
        )