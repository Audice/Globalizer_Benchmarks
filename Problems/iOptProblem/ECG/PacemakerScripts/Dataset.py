import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class SignalDataset(Dataset):
    def __init__(self, root_dir, split='train',
                 normalization_type='zscore',
                 use_differential=False,
                 augmentation=None,
                 target_length=5000):

        self.root_dir = root_dir
        self.split = split
        self.normalization_type = normalization_type
        self.use_differential = use_differential
        self.augmentation = augmentation
        self.target_length = target_length

        self.ecs_dir = os.path.join(root_dir, split, 'ecs')
        self.noecs_dir = os.path.join(root_dir, split, 'noecs')

        self.ecs_files = [os.path.join(self.ecs_dir, f) for f in os.listdir(self.ecs_dir) if f.endswith('.npy')]
        self.noecs_files = [os.path.join(self.noecs_dir, f) for f in os.listdir(self.noecs_dir) if f.endswith('.npy')]

        self.files = [(f, 1) for f in self.ecs_files] + [(f, 0) for f in self.noecs_files]

        if len(self.files) > 0:
            sample = np.load(self.files[0][0])
            self.n_channels = sample.shape[0]
            self.output_channels = self.n_channels * 2 if use_differential else self.n_channels
        else:
            self.n_channels = 12
            self.output_channels = 24 if use_differential else 12


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        signal = np.load(filepath).astype(np.float32)

        signal = self._fix_length(signal)

        signal = self._normalize_signal(signal)

        if self.use_differential:
            signal = self._add_differential_features(signal)

        # Аугментация
        if self.split == 'train' and self.augmentation:
            signal = self._apply_augmentation(signal)

        signal = torch.FloatTensor(signal).unsqueeze(0)
        return signal, torch.LongTensor([label])[0]

    def _fix_length(self, signal):
        current_length = signal.shape[1]
        if current_length == self.target_length:
            return signal
        elif current_length < self.target_length:
            pad_width = self.target_length - current_length
            return np.pad(signal, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            return signal[:, :self.target_length]

    def _normalize_signal(self, signal):
        if self.normalization_type == 'zscore':
            for ch in range(signal.shape[0]):
                mean = signal[ch].mean()
                std = signal[ch].std()
                if std > 1e-8:
                    signal[ch] = (signal[ch] - mean) / std
                else:
                    signal[ch] = signal[ch] - mean
        elif self.normalization_type == 'none':
            pass
        return signal

    def _add_differential_features(self, signal):
        diff1 = np.diff(signal, axis=1, prepend=signal[:, 0:1])
        for ch in range(diff1.shape[0]):
            std = diff1[ch].std()
            if std > 0:
                diff1[ch] = diff1[ch] / std
        return np.concatenate([signal, diff1], axis=0)

    def _apply_augmentation(self, signal):
        aug_type = np.random.choice(['none', 'noise', 'scale'])
        if aug_type == 'noise':
            noise_level = np.random.uniform(0.01, 0.05)
            signal += np.random.normal(0, noise_level * signal.std(), signal.shape)
        elif aug_type == 'scale':
            signal *= np.random.uniform(0.9, 1.1)
        return signal