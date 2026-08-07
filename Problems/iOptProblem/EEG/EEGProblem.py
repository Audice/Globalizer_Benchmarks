import warnings
import os
import sys

warnings.filterwarnings('ignore', category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trial import Point
from trial import FunctionValue
from problem import Problem
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
import torch
import torch.nn as nn


from dataclasses import dataclass, field
from typing import List, Dict, Any
from EEG.scripts.EEGClassificationDataset import *
from EEG.scripts.FuzzyPart import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from EEG.scripts.ThirdPartTools import *

DATA_DIR = '../dataset'
def Prepare(dataset_dir = '../dataset', batch_size = 32, num_workers = 0):
    train_dataset = EEGClassificationDataset(
        data_dir=DATA_DIR,
        mode='train',
        normalize=True,
        augment=True,
        target_length=2560,
        cache_data=True
    )

    val_dataset = EEGClassificationDataset(
        data_dir=DATA_DIR,
        mode='val',
        normalize=True,
        augment=False,
        target_length=2560,
        cache_data=True
    )

    test_dataset = EEGClassificationDataset(
        data_dir=DATA_DIR,
        mode='test',
        normalize=True,
        augment=False,
        target_length=2560,
        cache_data=True
    )

    # Вычисляем веса классов для сэмплинга
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    # Создаем сэмплер для балансировки
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader



class EEGProblem(Problem):
    def __init__(self, dimension: int = 6, ProcRank: int = 0):
        super(EEGProblem, self).__init__()

        print("Init ", dimension)
        self.dimension = dimension
        self.number_of_float_variables = 4
        self.number_of_discrete_variables = 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.train_loader, self.val_loader, self.test_loader = Prepare()


        self.float_variable_names = np.array(["focal_alpha", "cnn_embedding_dim", "lr_fuzzy", "backbone_feature_dim_options"], dtype=str)
        self.lower_bound_of_float_variables = [0.65, 32, 1e-3, 128]
        self.upper_bound_of_float_variables = [0.85, 128, 3e-2, 1024]


        self.discrete_variable_names = np.array(['num_rules', 'backbone_name'], dtype=str)
        self.discrete_variable_values([8, 12, 16, 20, 24, 32], ['simple', 'inception', 'resnet18', 'densenet'])


        GPU_count = torch.cuda.device_count()
        print(f"Доступно GPU: {GPU_count}")
        for i in range(GPU_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        self.gpu_id = 0
        if GPU_count != 0:
            self.gpu_id = ProcRank % GPU_count

            self.device = torch.device(f"cuda:{self.gpu_id}")

            torch.device(f"{self.device}")

        else:
            self.device = "cpu"

        print(f"gpu_id = {self.gpu_id}")

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:

        focal_alpha, cnn_embedding_dim, lr_fuzzy, backbone_feature_dim_options = point.float_variables[0], point.float_variables[1], point.float_variables[2], point.float_variables[3]

        num_rules, backbone_name = point.discrete_variables[0], point.discrete_variables[1]
        config = FuzzyCNNConfig(
            backbone_name=backbone_name,
            input_channels=23,
            input_width=2560,
            backbone_feature_dim=backbone_feature_dim_options,
            cnn_embedding_dim=cnn_embedding_dim,
            num_rules=num_rules,
            backbone_dropout=0.3,
            dropout_rate=0.3,
            projection_dropout=0.3,
            use_batch_norm=True,
            extra_params={} if backbone_name != 'densenet' else {'growth_rate': 32, 'n_layers': 4}
        )

        model = CNNWithFuzzy(config)

        trainer = FuzzyCNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            loss_type='focal',  # 'bce', 'weighted_bce', 'focal'
            loss_params={
                'alpha': focal_alpha,  # Вес для положительного класса
                'gamma': 2.0  # Фокусирующий параметр
            },
            device='cuda' if torch.cuda.is_available() else 'cpu',
            save_dir='checkpoints',
            experiment_name=f"{config.backbone_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # НАСТРОЙКА ОПТИМИЗАТОРА И ПЛАНИРОВЩИКА
        # Для бэкбона, слоя проекции и нечётких сетей свои learning_rate
        optimizer = torch.optim.Adam([
            {'params': model.backbone.parameters(), 'lr': 1e-3},
            {'params': model.projection.parameters(), 'lr': 1e-3},
            {'params': model.fuzzy_layer.parameters(), 'lr': lr_fuzzy}
        ], weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        trainer.fit(
            epochs=100,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping_patience=20,
            save_best=True,
            verbose=True
        )




        p, f = point.float_variables[0], point.float_variables[1]


        function_value.value = -trainer.best_val_mcc

        print('p ' + f"{p:.9f}" + '\tfeatures ' + f"{f:.9f}" + "\tvalue " + f"{function_value.value:.9f}", flush=True)
        return function_value


if __name__ == '__main__':
    problem_eeg_class = EEGProblem()