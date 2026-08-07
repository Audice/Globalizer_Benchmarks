from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from datetime import datetime, date, time, timedelta
from Metrics import ImbalancedMetrics
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import json


def compute_class_weights_from_folders(data_dir: str) -> torch.Tensor:
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'

    # Подсчет файлов в каждой папке
    epi_files = list((train_dir / 'epi').glob('*.npy'))
    norm_files = list((train_dir / 'norm').glob('*.npy'))

    n_epi = len(epi_files)
    n_norm = len(norm_files)
    total = n_epi + n_norm

    if n_epi == 0 or n_norm == 0:
        raise ValueError("У одного из классов нет представителей!")

    # Вычисляем веса
    weights = torch.tensor([
        total / (2 * n_norm),
        total / (2 * n_epi)
    ], dtype=torch.float32)

    # Нормализуем
    weights = weights / weights.sum() * 2

    #print(f"\nПредставители классов:")
    #print(f"  epi (class 1): {n_epi} samples ({n_epi / total * 100:.1f}%)")
    #print(f"  norm (class 0): {n_norm} samples ({n_norm / total * 100:.1f}%)")
    #print(f"  Веса классов: {weights.numpy()}")

    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: вес для положительного класса (0.75 для дисбаланса 90/10)
        gamma: фокусирующий параметр (2.0 по умолчанию)
        reduction: метод агрегации
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch, 1) - вероятности после sigmoid
        # targets: (batch,) - метки 0 или 1

        inputs = inputs.clamp(1e-7, 1 - 1e-7)
        targets = targets.float().view(-1, 1)

        # Вычисляем pt
        pt = torch.where(targets == 1, inputs, 1 - inputs)

        # Вычисляем alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal Loss
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """Взвешенная бинарная кросс-энтропия"""

    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, inputs, targets):
        inputs = inputs.clamp(1e-7, 1 - 1e-7)
        targets = targets.float().view(-1, 1)

        # Веса для каждого образца
        weights = torch.where(targets == 1, self.pos_weight, self.neg_weight)

        # BCE
        loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        loss = loss * weights

        return loss.mean()


def get_loss_function(loss_type='focal', **kwargs):
    """
    Фабрика функций потерь

    Args:
        loss_type: 'bce', 'weighted_bce', 'focal'
        **kwargs: параметры для конкретной функции
    """
    if loss_type == 'bce':
        return nn.BCELoss()

    elif loss_type == 'weighted_bce':
        pos_weight = kwargs.get('pos_weight', 9.0)  # Для дисбаланса 90/10
        neg_weight = kwargs.get('neg_weight', 1.0)
        return WeightedBCELoss(pos_weight, neg_weight)

    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha, gamma)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


"""
Тренер для CNN + Fuzzy
"""

class FuzzyCNNTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader = None,
                 loss_type: str = 'focal',
                 loss_params: dict = None,
                 device: str = None,
                 save_dir: str = 'checkpoints',
                 experiment_name: str = None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Функция потерь
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.criterion = get_loss_function(loss_type, **self.loss_params)

        # Метрики
        self.metrics = ImbalancedMetrics()

        # Директория для сохранения
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Имя эксперимента
        if experiment_name is None:
            self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name

        self.exp_dir = self.save_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'val_recall': [],
            'val_precision': [],
            'val_balanced_acc': [],
            'val_mcc': [],
            'learning_rates': [],
            'epochs': []
        }

        # Лучшие метрики
        self.best_val_f1 = 0.0
        self.best_val_mcc = 0.0
        self.best_epoch = 0

        print(f"✅ Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Loss: {loss_type}")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Save dir: {self.exp_dir}")
        print("=" * 60)

    def train_epoch(self, optimizer):
        """
        Обучает одну эпоху

        Returns:
            train_loss: средняя потеря за эпоху
            metrics: метрики на тренировочных данных
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device).squeeze()

            optimizer.zero_grad()

            # Forward
            outputs, rules, embeddings = self.model(data)
            outputs = outputs.squeeze()

            # Loss
            loss = self.criterion(outputs, labels.float())

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Сохраняем результаты
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            # Обновляем прогресс
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

        # Вычисляем метрики
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = self.metrics.calculate(all_preds, all_labels)

        return total_loss / len(self.train_loader), metrics

    def validate(self):
        """
        Валидация модели

        Returns:
            val_loss: средняя потеря на валидации
            metrics: метрики на валидационных данных
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc='Validation', leave=False):
                data = data.to(self.device)
                labels = labels.to(self.device).squeeze()

                outputs, rules, embeddings = self.model(data)
                outputs = outputs.squeeze()

                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = self.metrics.calculate_metrics(all_preds, all_labels)

        # Добавляем AUC
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_preds)
        except:
            metrics['auc'] = 0.0

        return total_loss / len(self.val_loader), metrics, all_preds, all_labels

    def test(self, test_loader=None):
        """
        Тестирование модели

        Args:
            test_loader: загрузчик тестовых данных (если не указан, использует сохраненный)

        Returns:
            metrics: метрики на тестовых данных
            predictions: предсказания
            labels: истинные метки
        """
        loader = test_loader or self.test_loader

        if loader is None:
            raise ValueError("Test loader not provided")

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in tqdm(loader, desc='Testing'):
                data = data.to(self.device)
                labels = labels.to(self.device).squeeze()

                outputs, rules, embeddings = self.model(data)
                outputs = outputs.squeeze()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = self.metrics.calculate(all_preds, all_labels)

        try:
            metrics['auc'] = roc_auc_score(all_labels, all_preds)
        except:
            metrics['auc'] = 0.0

        return metrics, all_preds, all_labels

    def fit(self,
            epochs: int = 100,
            optimizer: torch.optim.Optimizer = None,
            scheduler: torch.optim.lr_scheduler._LRScheduler = None,
            early_stopping_patience: int = 20,
            save_best: bool = True,
            verbose: bool = True):
        """
        Полный цикл обучения

        Args:
            epochs: количество эпох
            optimizer: оптимизатор (если None, создается Adam)
            scheduler: планировщик скорости обучения
            early_stopping_patience: количество эпох без улучшения для остановки
            save_best: сохранять ли лучшую модель
            verbose: печатать ли прогресс
        """

        # Создаем оптимизатор если не передан
        if optimizer is None:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-5
            )

        # Создаем планировщик если не передан
        if scheduler is None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=verbose
            )

        print("\n" + "=" * 60)
        print(f"🚀 STARTING TRAINING")
        print(f"   Epochs: {epochs}")
        print(f"   Loss: {self.loss_type}")
        print(f"   Device: {self.device}")
        print("=" * 60 + "\n")

        best_val_f1 = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Обучение
            train_loss, train_metrics = self.train_epoch(optimizer)

            # Валидация
            val_loss, val_metrics, val_preds, val_labels = self.validate()

            # Обновление планировщика
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            # Сохраняем историю
            current_lr = optimizer.param_groups[0]['lr']
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
            self.history['val_mcc'].append(val_metrics['mcc'])
            self.history['learning_rates'].append(current_lr)

            # Проверяем улучшение
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                self.best_val_f1 = best_val_f1
                self.best_val_mcc = val_metrics['mcc']
                self.best_epoch = epoch

                if save_best:
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1

            # Печатаем прогресс
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}")
                print(
                    f"  Val Loss:   {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, Recall: {val_metrics['recall']:.4f}, Prec: {val_metrics['precision']:.4f}")
                print(f"  LR: {current_lr:.2e}, Best F1: {best_val_f1:.4f}")
                print(f"  Patience: {patience_counter}/{early_stopping_patience}")

            # Сохраняем последний чекпоинт
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered after {epoch} epochs")
                break

        # Завершаем обучение
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ TRAINING COMPLETED")
        print(f"   Best F1: {self.best_val_f1:.4f} (epoch {self.best_epoch})")
        print(f"   Best MCC: {self.best_val_mcc:.4f}")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        print("=" * 60)

        # Сохраняем историю
        self.save_history()

        # Загружаем лучшую модель
        if save_best:
            self.load_checkpoint('best_model.pth')

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Сохраняет чекпоинт модели

        Args:
            epoch: номер эпохи
            metrics: словарь с метриками
            is_best: является ли чекпоинт лучшим
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': None,  # Не сохраняем для экономии места
            'metrics': metrics,
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'loss_type': self.loss_type,
            'loss_params': self.loss_params,
            'experiment_name': self.experiment_name
        }

        # Сохраняем чекпоинт
        if is_best:
            path = self.exp_dir / 'best_model.pth'
        else:
            path = self.exp_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_name='best_model.pth'):
        """
        Загружает чекпоинт модели

        Args:
            checkpoint_name: имя файла чекпоинта
        """
        path = self.exp_dir / checkpoint_name

        if not path.exists():
            print(f"⚠️ Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f"✅ Checkpoint loaded: {path}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   F1: {checkpoint['metrics']['f1']:.4f}")

    def save_history(self):
        """Сохраняет историю обучения в JSON"""
        history_path = self.exp_dir / 'history.json'
        with open(history_path, 'w') as f:
            # Преобразуем numpy массивы в списки
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list):
                    history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
                else:
                    history_serializable[key] = value
            json.dump(history_serializable, f, indent=2)

        # Сохраняем как DataFrame
        df_history = pd.DataFrame(self.history)
        df_history.to_csv(self.exp_dir / 'history.csv', index=False)


    def evaluate_on_test(self, test_loader=None, threshold=0.5):
        """
        Оценка модели на тестовых данных

        Args:
            test_loader: загрузчик тестовых данных
            threshold: порог бинаризации

        Returns:
            dict: результаты
        """
        loader = test_loader or self.test_loader

        if loader is None:
            raise ValueError("Test loader not provided")

        # Загружаем лучшую модель
        self.load_checkpoint('best_model.pth')

        # Тестирование
        metrics, preds, labels = self.test(loader)

        # Визуализация результатов
        self.plot_test_results(preds, labels, threshold)

        # Сохраняем результаты
        results = {
            'metrics': metrics,
            'predictions': preds,
            'labels': labels
        }

        np.save(self.exp_dir / 'test_predictions.npy', preds)
        np.save(self.exp_dir / 'test_labels.npy', labels)

        # Сохраняем отчет
        with open(self.exp_dir / 'test_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print("📊 TEST RESULTS")
        print("=" * 60)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision:          {metrics['precision']:.4f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"Specificity:        {metrics['specificity']:.4f}")
        print(f"F1 Score:           {metrics['f1']:.4f}")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"MCC:                {metrics['mcc']:.4f}")
        print(f"AUC:                {metrics.get('auc', 0):.4f}")
        print("=" * 60)
        print(f"TP: {metrics['tp']}, TN: {metrics['tn']}")
        print(f"FP: {metrics['fp']}, FN: {metrics['fn']}")
        print("=" * 60)

        return results

    def get_summary(self):
        """
        Возвращает сводку по обучению

        Returns:
            dict: сводка
        """
        return {
            'experiment_name': self.experiment_name,
            'best_val_f1': self.best_val_f1,
            'best_val_mcc': self.best_val_mcc,
            'best_epoch': self.best_epoch,
            'loss_type': self.loss_type,
            'loss_params': self.loss_params,
            'num_epochs': len(self.history['epochs']),
            'device': str(self.device),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'save_dir': str(self.exp_dir),
        }

    def print_summary(self):
        """Печатает сводку"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("📊 TRAINING SUMMARY")
        print("=" * 60)
        for key, value in summary.items():
            print(f"   {key}: {value}")
        print("=" * 60)
