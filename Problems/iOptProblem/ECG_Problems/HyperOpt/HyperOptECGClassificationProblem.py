import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import json
import uuid
from ClassificationScripts.dataset import ECGDataset
from ClassificationScripts.model import MobileNetV3Small1D
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import sys
from urllib.parse import urlencode
import requests
import zipfile
import os

# HyperOpt imports
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from trial import Point
from trial import FunctionValue
from problem import Problem
from Globalizer_problem import GlobalizerProblem


def downloadClassificationDataset():
    path = Path(__file__).parent
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/2G-w_EXbKFb7QA'
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    download_response = requests.get(download_url)
    with open(path / 'datasets.zip', 'wb') as f:
        f.write(download_response.content)
    Path(path / 'datasets').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path / 'datasets.zip') as f:
        f.extractall(path=(path / 'dataset' / 'ECGClassification'))
    os.remove(path / 'datasets.zip')


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, gpu_id=-1, patience=100):
    if gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_f1, best_state


def evaluate(model, data_loader, device, f1_average='macro'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return f1_score(all_labels, all_preds, average=f1_average)


def prepare_data(base_path):
    X = []
    y = np.load(base_path / "Y.npy")
    for i in range(len(y)):
        X.append(np.load(base_path / "X" / (str(i) + ".npy")))
    X = np.array(X)
    X = X[:, :, 0:2]
    y = np.array(y)
    return X, y


class HyperoptECGClassification(Problem):
    def __init__(self, max_evals: int = 100, timeout: int = None):
        super().__init__()
        self.N_FOLDS = 5
        self.base_path = Path("dataset/ECGClassification")
        X, y = prepare_data(self.base_path)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        X_trainval = np.moveaxis(X_trainval, [1], [2])
        X_test = np.moveaxis(X_test, [1], [2])

        skf = StratifiedKFold(n_splits=self.N_FOLDS, shuffle=True, random_state=42)
        self.folds: List[Tuple[DataLoader, DataLoader]] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval)):
            X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
            y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
            train_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=32, shuffle=True)
            val_loader = DataLoader(ECGDataset(X_va, y_va), batch_size=32, shuffle=False)
            self.folds.append((train_loader, val_loader))

        self.full_train_loader = DataLoader(
            ECGDataset(X_trainval, y_trainval), batch_size=32, shuffle=True
        )
        self.test_loader = DataLoader(
            ECGDataset(X_test, y_test), batch_size=32, shuffle=False
        )

        self.model_save_path = Path("models")
        if not Path.exists(self.model_save_path):
            Path.mkdir(self.model_save_path, parents=True, exist_ok=True)

        self.dimension = 2
        self.number_of_float_variables = 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.float_variable_names = np.array(["P parameter", "Features parameter"], dtype=str)
        self.lower_bound_of_float_variables = [0.0, 80.0]
        self.upper_bound_of_float_variables = [1.0, 200.0]

        self.max_evals = max_evals
        self.timeout = timeout
        self.best_params = None
        self.best_value = None
        self.trials = None

        GPU_count = torch.cuda.device_count()
        print(f"Доступно GPU: {GPU_count}")
        for i in range(GPU_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
            
        self.gpu_id = -1
        if GPU_count != 0:
            self.gpu_id = 0
            self.device = torch.device(f"cuda:{self.gpu_id}")
            torch.device(f"{self.device}")
        else:
            self.device = "cpu"

        print(f"gpu_id = {self.gpu_id}")
        

    def _build_model(self, p: float, f: float) -> nn.Module:
        return MobileNetV3Small1D(
            in_channels=2, num_classes=3, p=p, o_features=int(f)
        )

    def objective(self, params: dict) -> dict:
        """Целевая функция для HyperOpt. Возвращает словарь с loss и status."""
        p = params['p']
        f = params['f']
        model_params = {"p": float(p), "f": float(f)}

        f1_scores: List[float] = []
        best_state_overall = None
        best_f1_overall = -1.0

        for fold_idx, (train_loader, val_loader) in enumerate(self.folds):
            model = self._build_model(p, f)
            best_val_f1, best_state = train(
                model, train_loader, val_loader,
                epochs=100, lr=1e-3, gpu_id=self.gpu_id, patience=20
            )
            f1_scores.append(best_val_f1)
            if best_val_f1 > best_f1_overall:
                best_f1_overall = best_val_f1
                best_state_overall = best_state
            print(f"  fold {fold_idx + 1}/{self.N_FOLDS}: val_f1={best_val_f1:.4f}")

        mean_f1 = float(np.mean(f1_scores))
        if best_state_overall is not None:
            model = self._build_model(p, f)
            model.load_state_dict(best_state_overall)

        # HyperOpt минимизирует, поэтому возвращаем отрицательный F1
        loss = -mean_f1
        self.save_model(model, loss, model_params)

        # Логирование в файл (аналог save_trial_callback)
        with open("classification_hyperopt_log.txt", "a", encoding="utf-8") as log_f:
            log_f.write("=" * 60 + "\n")
            log_f.write(f"Params: p={p:.6f}, f={f:.2f}\n")
            log_f.write(f"Loss (-mean_f1): {loss:.6f}\n")
            log_f.write(f"Mean F1: {mean_f1:.6f}\n")
            log_f.write("=" * 60 + "\n\n")

        return {'loss': loss, 'status': STATUS_OK}

    def save_model(self, model, metric, params_dict):
        files = [f for f in self.model_save_path.iterdir() if f.is_file()]
        for f in files:
            f_metric = float(f.stem.split("_")[0])
            if f_metric < metric:
                return
        print("Модель сохранена: ", metric, params_dict)
        proc_id = uuid.uuid4().hex[:8]
        local_path = self.model_save_path.joinpath(f"{metric}_{proc_id}.pth")
        torch.save(model.state_dict(), local_path)
        with open(self.model_save_path.joinpath(f"{metric}_{proc_id}.json"), "w", encoding="utf-8") as f:
            json.dump(params_dict, f, ensure_ascii=False, indent=2)

    def match_by_fields(self, d1, d2, fields):
        return all(d1.get(f) == d2.get(f) for f in fields)

    def finalize(self, coordinate: List[float], discreteCoordinate: List[str]) -> float:
        print("Проверка тестовой выборки в лучшей точке: ", coordinate, discreteCoordinate)
        p, f = coordinate[0], coordinate[1]
        tmp_params_dict = {'p': p, 'f': f}
        model_params = None
        metric = None
        model_path = None

        json_files = list(Path(self.model_save_path).glob("*.json"))
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f_read:
                current_params_dict = json.load(f_read)
                if self.match_by_fields(current_params_dict, tmp_params_dict, ['p', 'f']):
                    model_params = current_params_dict
                    metric = float(json_file.stem.split("_")[0])
                    model_path = json_file.stem
                    break

        if model_params is None:
            print("Среди чекпоинтов не найден целевой!")
            return -1

        target_path = self.model_save_path.joinpath(f"{model_path}.pth")
        print("Лучшая модель: ", target_path)

        model = self._build_model(model_params["p"], model_params["f"])
        model.load_state_dict(torch.load(target_path, map_location=self.device))
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(100):
            for X_batch, y_batch in self.full_train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), self.model_save_path / f'{finish_metric}_final_model_{uuid.uuid4().hex[:8]}.pth')
        finish_metric = evaluate(model, self.test_loader, self.device)
        print(f'mean cross-validated macro-F1 по {self.N_FOLDS} фолдам: ', abs(metric))
        print("Значение целевой метрики на тестовой выборке: ", finish_metric)
        return finish_metric

    def optimize_with_hyperopt(self) -> None:
        """Запуск оптимизации через HyperOpt."""
        # Определяем пространство поиска HyperOpt
        # hp.uniform(label, low, high) задаёт равномерное распределение
        space = {
            'p': hp.uniform('p', self.lower_bound_of_float_variables[0],
                            self.upper_bound_of_float_variables[0]),
            'f': hp.uniform('f', self.lower_bound_of_float_variables[1],
                            self.upper_bound_of_float_variables[1])
        }

        self.trials = Trials()

        # fmin — основная функция HyperOpt
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,  # Tree of Parzen Estimators (байесовская оптимизация)
            max_evals=self.max_evals,
            trials=self.trials,
            timeout=self.timeout if self.timeout else None,
            show_progressbar=True
        )

        self.best_params = best
        # Восстанавливаем положительное значение F1 из loss
        self.best_value = -min(t['result']['loss'] for t in self.trials.trials if t['result']['loss'] is not None)

        print(f"Лучшее значение целевой функции (mean F1): {self.best_value:.4f}")
        print("\nЛучшие гиперпараметры:")
        print(f"  p = {best['p']}")
        print(f"  f = {best['f']}")

        # Финальная проверка на тестовой выборке
        test_val = self.finalize([best['p'], best['f']], [])


if __name__ == '__main__':
    downloadClassificationDataset()

    with open("classification_hyperopt_log.txt", "a", encoding="utf-8") as f:
        current_datetime = datetime.now()
        f.write("=" * 60 + "\n")
        f.write(f"Start {current_datetime}:\n")
        f.write("=" * 60 + "\n\n")

    optimizer = HyperoptECGClassification(
        max_evals=320,  # Количество испытаний (аналог n_trials в Optuna)
        timeout=10 * 3600  # Максимальное время в секундах
    )
    optimizer.optimize_with_hyperopt()

    with open("classification_hyperopt_log.txt", "a", encoding="utf-8") as f:
        current_datetime = datetime.now()
        f.write("=" * 60 + "\n")
        f.write(f"Finish {current_datetime}:\n")
        f.write("=" * 60 + "\n\n")
