"""
Создает загрузчики данных из структуры папок с классами
"""

from scripts import EEGClassificationDataset as EEGDataset
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict

def create_dataloaders_from_folders(data_dir: str,
                                    batch_size: int = 32,
                                    num_workers: int = 0,
                                    normalize: bool = True,
                                    augment: bool = True,
                                    target_length: Optional[int] = 2560,
                                    cache_data: bool = True) -> Dict[str, DataLoader]:

    data_dir = Path(data_dir)

    # Проверяем наличие подпапок
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    # Проверяем структуру
    required_dirs = ['epi', 'norm']
    for dir_name in required_dirs:
        if not (train_dir / dir_name).exists():
            raise FileNotFoundError(f"Директория с датасетами {train_dir / dir_name} не найдена!")

    # Создаем датасеты
    print("\n" + "=" * 60)
    print("Загрузка...")
    print("=" * 60)

    train_dataset = EEGDataset.EEGClassificationDataset(
        data_dir=data_dir,
        mode='train',
        normalize=normalize,
        augment=augment,
        target_length=target_length,
        cache_data=cache_data
    )

    val_dataset = EEGDataset.EEGClassificationDataset(
        data_dir=data_dir,
        mode='val',
        normalize=normalize,
        augment=False,
        target_length=target_length,
        cache_data=cache_data
    ) if val_dir.exists() else None

    if test_dir.exists():
        test_dataset = EEGDataset.EEGClassificationDataset(
            data_dir=data_dir,
            mode='test',
            normalize=normalize,
            augment=False,
            target_length=target_length,
            cache_data=cache_data
        )
    else:
        test_dataset = None

    # Создаем загрузчики
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if val_dataset is not None else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if test_dataset is not None else None

    print("Даталоадеры созданы!")
    print(f"Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    if val_loader:
        print(f"Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    if test_loader:
        print(f"Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }