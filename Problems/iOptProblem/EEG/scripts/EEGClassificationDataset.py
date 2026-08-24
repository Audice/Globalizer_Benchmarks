from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import torch


class EEGClassificationDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 mode: str = 'train',  # 'train', 'val', 'test'
                 normalize: bool = True,
                 scaler: Optional[StandardScaler] = None,
                 augment: bool = False,
                 target_length: Optional[int] = None,
                 cache_data: bool = True):

        self.data_dir = Path(data_dir)
        self.mode = mode
        self.normalize = normalize
        self.augment = augment and (mode == 'train')
        self.target_length = target_length
        self.cache_data = cache_data

        # Путь к папке с данными для текущего режима
        self.mode_dir = self.data_dir / mode

        if not self.mode_dir.exists():
            raise FileNotFoundError(f"Папка {self.mode_dir} не найдена!")

        self.files = []
        self.labels = []

        self.class_map = {'epi': 1, 'norm': 0}

        for class_name, label in self.class_map.items():
            class_dir = self.mode_dir / class_name
            if class_dir.exists():
                npy_files = list(class_dir.glob('*.npy'))
                if not npy_files:
                    print(f"Файлы сигналов не найдены в директории {class_dir}")
                for file_path in npy_files:
                    self.files.append(file_path)
                    self.labels.append(label)

        if not self.files:
            raise ValueError(f"Данные не найдены в {self.mode_dir}/[epi,norm]")

        print(f"Представители классов: epi={sum(1 for l in self.labels if l == 1)}, norm={sum(1 for l in self.labels if l == 0)}")

        if self.cache_data:
            self.data_cache = []
            print(f"Загрузка данных в память...")
            for file_path in tqdm(self.files):
                data = np.load(file_path)
                if self.target_length is not None:
                    data = self._resize_signal(data, self.target_length)
                self.data_cache.append(data)
        else:
            self.data_cache = None

        if self.normalize:
            if mode == 'train':
                self._fit_scaler()
            else:
                self._load_scaler()

    def _fit_scaler(self):
        all_data = []
        for file_path in tqdm(self.files):
            data = np.load(file_path)
            if self.target_length is not None:
                data = self._resize_signal(data, self.target_length)
            all_data.append(data.reshape(-1))

        all_data = np.array(all_data)
        self.scaler = StandardScaler()
        self.scaler.fit(all_data.reshape(-1, 1))

        # Сохраняем scaler
        scaler_path = self.data_dir / f'scaler_{self.mode}.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler сохранён {scaler_path}")

    def _load_scaler(self):
        scaler_path = self.data_dir / 'scaler_train.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler не найден по пути {scaler_path}.")
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler загружен {scaler_path}")

    def _resize_signal(self, data: np.ndarray, target_length: int) -> np.ndarray:
        current_length = data.shape[-1]

        if current_length == target_length:
            return data

        if current_length > target_length:
            start = (current_length - target_length) // 2
            return data[..., start:start + target_length]
        else:
            pad_left = (target_length - current_length) // 2
            pad_right = target_length - current_length - pad_left
            pad_width = [(0, 0)] * (data.ndim - 1) + [(pad_left, pad_right)]
            return np.pad(data, pad_width, mode='constant')

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        flat_data_norm = self.scaler.transform(flat_data)
        return flat_data_norm.reshape(original_shape)

    def _augment_signal(self, data: np.ndarray) -> np.ndarray:
        augmented = data.copy()
        time_len = augmented.shape[-1]

        if np.random.random() > 0.5:
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented += noise

        if np.random.random() > 0.5:
            shift = np.random.randint(-time_len // 20, time_len // 20)
            if shift != 0:
                augmented = np.roll(augmented, shift, axis=-1)
                if shift > 0:
                    augmented[..., :shift] = 0
                else:
                    augmented[..., shift:] = 0

        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            augmented *= scale

        if np.random.random() > 0.9:
            augmented = -augmented

        return augmented

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_data and self.data_cache is not None:
            data = self.data_cache[idx].copy()
        else:
            data = np.load(self.files[idx])
            if self.target_length is not None:
                data = self._resize_signal(data, self.target_length)

        label = self.labels[idx]

        if self.normalize:
            data = self._normalize_data(data)

        if self.augment:
            data = self._augment_signal(data)

        if data.ndim == 2:
            data_tensor = torch.FloatTensor(data)
        elif data.ndim == 1:
            data_tensor = torch.FloatTensor(data).unsqueeze(0)
        else:
            data_tensor = torch.FloatTensor(data)
        label_tensor = torch.FloatTensor([label])
        return data_tensor, label_tensor