import warnings
from typing import List
import sys
warnings.filterwarnings('ignore', category=UserWarning)
from ClassificationScripts.dataset import ECGDataset
from ClassificationScripts.model import MobileNetV3Small1D
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import json

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from trial import Point
from trial import FunctionValue
from problem import Problem
from Globalizer_problem import GlobalizerProblem

def train(model, train_loader, val_loader, epochs=10, lr=1e-3, gpu_id=0, patience=100):
    if gpu_id != 0:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device('cpu')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_f1_scores = []
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        val_f1 = evaluate(model, val_loader, device)
        val_f1_scores.append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_val_f1


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
    y = np.load(base_path / "Y.npy") # np.load(Path("datasets/ECGClassification/Y.npy"))

    for i in range(len(y)):
        X.append(np.load(base_path / "X"/(str(i) + ".npy")))

    X = np.array(X)
    X = X[:, :, 0:2]
    y = np.array(y)
    return X, y

class ECGClassificationProblem(Problem):
    def __init__(self, dimension: int, ProcRank:int=0):
        super(ECGClassificationProblem, self).__init__()
        print("Init ", dimension)
        self.dimension = dimension
        self.number_of_float_variables = dimension
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        print(dimension, ProcRank)
        self.base_path = Path("dataset/ECGClassification")
        X, y = prepare_data(self.base_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        X_train = np.moveaxis(X_train, [1], [2])
        #X_test = np.moveaxis(X_test, [1], [2])

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

        X_val = np.moveaxis(X_val, [1], [2])

        X_test = np.moveaxis(X_test, [1], [2])



        train_dataset = ECGDataset(X_train, y_train)
        test_dataset = ECGDataset(X_test, y_test)
        val_dataset = ECGDataset(X_val, y_val)

        self.model_save_path = Path("models")
        if not Path.exists(self.model_save_path):
            Path.mkdir(self.model_save_path)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32)
        self.val_loader = DataLoader(val_dataset, batch_size=32)


        self.float_variable_names = np.array(["P parameter", "Features parameter"], dtype=str)
        self.lower_bound_of_float_variables = [0.0, 80.0]
        self.upper_bound_of_float_variables = [1.0, 200.0]

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
        p, f = point.float_variables[0], point.float_variables[1]

        model_params = {
            'p': p,
            'f': f
        }
        model = MobileNetV3Small1D(in_channels=2, num_classes=3, p=p, o_features=int(f))
        macro_F1 = train(model, self.train_loader, self.val_loader, epochs=100, lr=1e-3, gpu_id=self.gpu_id)

        function_value.value = -macro_F1

        self.save_model(model, function_value.value, model_params)

        return function_value

    def save_model(self, model, metric, params_dict):
        files = [f for f in self.model_save_path.iterdir() if f.is_file()]

        for f in files:
            f_metric = float(f.stem)
            if f_metric < metric:
                return

        print("Модель сохранена: ", metric, params_dict)
        local_path = self.model_save_path.joinpath(f"{metric}.pth")
        torch.save(model.state_dict(), local_path)
        with open(self.model_save_path.joinpath(f"{metric}.json"), "w", encoding="utf-8") as f:
            json.dump(params_dict, f, ensure_ascii=False, indent=2)


    def match_by_fields(self, d1, d2, fields):
        return all(d1.get(f) == d2.get(f) for f in fields)

    def finalize(self, coordinate: List[float], discreteCoordinate: List[str]) -> float:
        """
        The process that occurs after the solution search procedure
        return: Calculated values of target metric"""
        
        print("Проверка тестовой выборки в лучшей точке: ", coordinate, discreteCoordinate)
        p, f = coordinate[0], coordinate[1]

        model_params = None

        tmp_params_dict = {
            'p': p,
            'f': f
        }


        json_files = list(Path(self.model_save_path).glob("*.json"))
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                current_params_dict = json.load(f)
                if self.match_by_fields(current_params_dict, tmp_params_dict,
                                        ['p', 'f']):
                    model_params = current_params_dict
                    metric = float(json_file.stem)
                    break

        if model_params is None:
            print("Среди чекпоинтов не найден целевой!")
            return -1

        target_path = self.model_save_path.joinpath(f"{metric}.pth")
        print("Лучшая модель: ", target_path)

        model = MobileNetV3Small1D(in_channels=2, num_classes=3, p=model_params['p'], o_features=int(model_params['f']))
        model.load_state_dict(torch.load(target_path, map_location=self.device))
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = model(X_batch)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        finish_metric = f1_score(all_labels, all_preds, average='macro')
        print("Значение целевой метрики на валидационной выборке: ", metric)
        print("Значение целевой метрики на тестовой выборке: ", finish_metric)

        return finish_metric


if __name__ == '__main__':
    problem_ecg_class = ECGClassificationProblem(2)
    problem = GlobalizerProblem(problem_ecg_class)
    result = problem.calculate([0.3, 100], [])
    problem.finalize([0.3, 100], [])
    print(result)