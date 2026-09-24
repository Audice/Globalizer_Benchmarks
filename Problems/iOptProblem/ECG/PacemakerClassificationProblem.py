import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, balanced_accuracy_score
from tqdm import tqdm
import json
from trial import Point
from trial import FunctionValue
from problem import Problem
from ECG.PacemakerScripts.Dataset import *
from ECG.PacemakerScripts.Model import *
from ECG.PacemakerScripts.MetricsCalculator import *
from ECG.PacemakerScripts.Trainer import *
import uuid
import warnings
warnings.filterwarnings('ignore')


class PacemakerClassificationProblem(Problem):
    def __init__(self, dimension: int=8, ProcRank:int=0):
        super(PacemakerClassificationProblem, self).__init__()
        print("Init ", dimension)
        self.dimension = dimension
        self.number_of_float_variables = dimension
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        print(dimension, ProcRank)

        torch.manual_seed(42)
        np.random.seed(42)

        dataset_path = './dataset'
        train_dataset = SignalDataset(
            dataset_path, 'train',
            normalization_type='zscore',
            use_differential=False,
            augmentation=True,
            target_length=5000
        )
        val_dataset = SignalDataset(
            dataset_path, 'val',
            normalization_type='zscore',
            use_differential=False,
            augmentation=False,
            target_length=5000
        )
        test_dataset = SignalDataset(
            dataset_path, 'test',
            normalization_type='zscore',
            use_differential=False,
            augmentation=False,
            target_length=5000
        )

        self.class_counts = [len(train_dataset.noecs_files), len(train_dataset.ecs_files)]
        print(f"\nClass counts - NOECS: {self.class_counts[0]}, ECS: {self.class_counts[1]}")

        class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float)
        sample_weights = [class_weights[label] for _, label in train_dataset.files]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        batch_size = 128

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)






        self.float_variable_names = np.array(['num_blocks', 'growth_rate', 'dropout', 'learning_rate',
                                              'weight_decay', 'loss_alpha', 'loss_gamma', 'kernel_size'], dtype=str)
        self.lower_bound_of_float_variables = [2, 16, 0.2, 0.0001, 0.00005, 0.5, 1.0, 3]
        self.upper_bound_of_float_variables = [4, 64, 0.7, 0.002, 0.001, 0.9, 5.0, 9]


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
        num_blocks, growth_rate, dropout, learning_rate, weight_decay, loss_alpha, loss_gamma, kernel_size =  int(point.float_variables[0]), int(point.float_variables[1]), point.float_variables[2], point.float_variables[3],point.float_variables[4], point.float_variables[5], point.float_variables[6], int(point.float_variables[7])

        model_params = {
            'input_channels': 12,
            'num_blocks': num_blocks,
            'growth_rate': growth_rate,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'use_batch_norm': True,
            'use_se': False
        }


        model = Xception1D(**model_params).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        criterion = WeightedFocalLoss(
            self.class_counts,
            alpha=loss_alpha,
            gamma=loss_gamma,
            beta=0.999
        ).to(self.device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=20, factor=0.5
        )

        save_dir = './xception'
        os.makedirs(save_dir, exist_ok=True)
        trainer = Trainer(model, self.device, uuid.uuid4(), save_dir)

        history, best_target_metric = trainer.train(
            self.train_loader, self.val_loader, criterion, optimizer,
            epochs=100,
            patience=20,
            scheduler=scheduler
        )

        function_value.value = -best_target_metric

        return function_value