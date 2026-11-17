import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, List
from sklearn.utils import shuffle
import csv
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from urllib.parse import urlencode
import requests
import zipfile
import os
import optuna
import warnings
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from trial import Point
from trial import FunctionValue
from problem import Problem
from Globalizer_problem import GlobalizerProblem

warnings.filterwarnings('ignore')


def factory_dataset():
    path = Path(__file__).parent    
    x = []
    y = []
    with open(path / 'transformator_state.csv') as rrrr_file:
        file_reader = csv.reader(rrrr_file, delimiter=",")
        for row in file_reader:
            x_row = []
            for i in range(len(row)-1):
                x_row.append(row[i])
            x.append(x_row)
            y.append(row[len(row)-1])
    return shuffle(np.array(x), np.array(y), random_state=42)


def downloadTransformatorsDataset():
    path = Path(__file__).parent

    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/sQKxgc9GNfM1nw'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    download_response = requests.get(download_url)
    with open(path / 'transformator_state.csv', 'wb') as f:
        f.write(download_response.content)


class SVC_3D_Optuna(Problem):
    def __init__(self, n_trials: int = 100, timeout: int = None):
        super().__init__()

        # Загрузка данных
        x_dataset, y_dataset = factory_dataset()

        self.dimension = 3
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')

        self.x = x_dataset
        self.y = y_dataset
        self.float_variable_names = ["Regularization parameter", "Kernel coefficient"]
        self.lower_bound_of_float_variables = [5.0, -3.0]
        self.upper_bound_of_float_variables = [9.0, 1.0]
        self.discrete_variable_names.append('kernel')
        self.discrete_variable_values.append(('sigmoid', 'rbf', 'poly'))
        self.cv = StratifiedKFold(shuffle=True, random_state=42)

        # Параметры для Optuna
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None
        self.best_value = None

    def objective(self, trial: optuna.Trial) -> float:
        """Целевая функция для Optuna"""
        # Предлагаем значения гиперпараметров
        cs = trial.suggest_float('C_log',
                                 self.lower_bound_of_float_variables[0],
                                 self.upper_bound_of_float_variables[0])
        gamma_log = trial.suggest_float('gamma_log',
                                        self.lower_bound_of_float_variables[1],
                                        self.upper_bound_of_float_variables[1])
        kernel_type = trial.suggest_categorical('kernel', ['sigmoid', 'rbf', 'poly'])

        # Преобразуем из логарифмической шкалы
        C = 10 ** cs
        gamma = 10 ** gamma_log

        # Создаем и обучаем модель
        clf = SVC(C=C, gamma=gamma, kernel=kernel_type, random_state=42)

        # Вычисляем метрику через кросс-валидацию
        f1_scores = cross_val_score(clf, self.x, self.y, cv=self.cv, scoring='f1_macro')
        f1_mean = f1_scores.mean()

        # Optuna минимизирует функцию, поэтому возвращаем 1 - f1
        return 1 - f1_mean

    def optimize_with_optuna(self, show_plots: bool = True) -> None:
        #"""Оптимизация гиперпараметров с помощью Optuna"""
        #print("=" * 60)
        #print("Начинаем оптимизацию гиперпараметров SVM с помощью Optuna")
        #print(f"Количество trials: {self.n_trials}")
        #print("=" * 60)

        # Создаем исследование
        self.study = optuna.create_study(
            study_name='svm_hyperparameter_optimization',
            direction='minimize',  # Минимизируем 1 - f1_macro
            sampler=optuna.samplers.TPESampler(seed=42),  # Детерминированный семплинг
            pruner=optuna.pruners.MedianPruner(),  # Отсечение бесперспективных trials
            storage=None  # Можно указать БД для сохранения: 'sqlite:///svm_study.db'
        )

        # Запускаем оптимизацию
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Сохраняем лучшие результаты
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Выводим результаты
        #print("\n" + "=" * 60)
        #print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        #print("=" * 60)
        print(f"Лучшее значение целевой функции : {self.best_value:.4f}")
        #print(f"Соответствующее значение f1_macro: {1 - self.best_value:.4f}")
        print("\nЛучшие гиперпараметры:")
        for param, value in self.best_params.items():
            if param == 'C_log':
                print(f"  C (10^{value:.2f}): {10 ** value:.2e}")
            elif param == 'gamma_log':
                print(f"  gamma (10^{value:.2f}): {10 ** value:.2e}")
            else:
                print(f"  {param}: {value}")
        
        ## Визуализация результатов
        #if show_plots:
        #    try:
        #        import plotly
        #        self.plot_results()
        #    except ImportError:
        #        print("\nДля визуализации установите plotly: pip install plotly")

    def plot_results(self) -> None:
        """Визуализация результатов оптимизации"""
        import plotly.io as pio
        pio.renderers.default = "browser"  # или "notebook" для Jupyter

        # История оптимизации
        fig1 = optuna.visualization.plot_optimization_history(self.study)
        fig1.show()

        # Важность параметров
        fig2 = optuna.visualization.plot_param_importances(self.study)
        fig2.show()

        # Распределение параметров
        fig3 = optuna.visualization.plot_parallel_coordinate(self.study)
        fig3.show()

        # Контурный график для C и gamma
        fig4 = optuna.visualization.plot_contour(
            self.study,
            params=['C_log', 'gamma_log']
        )
        fig4.show()

    def get_best_model(self) -> SVC:
        """Возвращает модель с лучшими найденными параметрами"""
        if self.best_params is None:
            raise ValueError("Сначала выполните оптимизацию с помощью optimize_with_optuna()")

        # Преобразуем из логарифмической шкалы
        C = 10 ** self.best_params['C_log']
        gamma = 10 ** self.best_params['gamma_log']
        kernel = self.best_params['kernel']

        return SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            random_state=42,
            probability=True  # Для возможности получать вероятности
        )

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """Оригинальный метод для совместимости с существующим кодом"""
        cs, gammas = point.float_variables[0], point.float_variables[1]
        kernel_type = point.discrete_variables[0]

        clf = SVC(C=10 ** cs, gamma=10 ** gammas, kernel=kernel_type, random_state=42)
        f1_mean = cross_val_score(clf, self.x, self.y, cv=self.cv, scoring='f1_macro').mean()
        function_value.value = 1 - f1_mean

        return function_value

    def calculate_with_optuna_params(self) -> float:
        """Расчет с лучшими параметрами из Optuna"""
        if self.best_params is None:
            self.optimize_with_optuna(show_plots=False)

        model = self.get_best_model()
        f1_scores = cross_val_score(model, self.x, self.y, cv=self.cv, scoring='f1_macro')
        return f1_scores.mean()

    def default_calculate(self) -> float:
        """Метод расчёта значения целевой функции со значениями по умолчанию"""
        clf = SVC(random_state=42)
        value = 1 - cross_val_score(clf, self.x, self.y, cv=self.cv, scoring='f1_macro').mean()
        return value

    def compare_with_default(self) -> None:
        """Сравнение производительности с моделью по умолчанию"""
        default_f1 = 1 - self.default_calculate()
        optimized_f1 = self.best_value if self.best_value is not None else self.calculate_with_optuna_params()

        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ С МОДЕЛЬЮ ПО УМОЛЧАНИЮ")
        print("=" * 60)
        print(f"F1-macro по умолчанию: {1 - default_f1:.4f}")
        print(f"F1-macro после оптимизации: {1 - optimized_f1:.4f}")
        improvement = ((1 - optimized_f1) - (1 - default_f1)) / (1 - default_f1) * 100
        print(f"Улучшение: {improvement:+.2f}%")


class SVC_3D(Problem):
    """Оригинальный класс для обратной совместимости"""

    def __init__(self):
        super().__init__()
        x_dataset, y_dataset = factory_dataset()

        self.dimension = 3
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 1
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        if x_dataset.shape[0] != y_dataset.shape[0]:
            raise ValueError('The input and output sample sizes do not match.')
        self.x = x_dataset
        self.y = y_dataset
        self.float_variable_names = ["Regularization parameter", "Kernel coefficient"]
        self.lower_bound_of_float_variables = [5.0, -3.0]
        self.upper_bound_of_float_variables = [9.0, 1.0]
        self.discrete_variable_names.append('kernel')
        self.discrete_variable_values.append(('sigmoid', 'rbf', 'poly'))
        self.cv = StratifiedKFold(shuffle=True, random_state=42)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        cs, gammas = point.float_variables[0], point.float_variables[1]
        kernel_type = point.discrete_variables[0]
        clf = SVC(C=10 ** cs, gamma=10 ** gammas, kernel=kernel_type)
        function_value.value = 1 - cross_val_score(clf, self.x, self.y, cv=self.cv, scoring='f1_macro').mean()
        return function_value

    def default_calculate(self) -> float:
        clf = SVC()
        value = 1 - cross_val_score(clf, self.x, self.y, cv=self.cv, scoring='f1_macro').mean()
        return value


if __name__ == '__main__':
    # Загружаем данные
    downloadTransformatorsDataset()

    # Создаем оптимизатор
    svm_optimizer = SVC_3D_Optuna(
        n_trials=36,  # Количество испытаний
        timeout=3600  # Максимальное время в секундах (1 час)
    )

    # Выполняем оптимизацию
    svm_optimizer.optimize_with_optuna(show_plots=True)
    
    best_params_transformed = {
        'C': 10 ** svm_optimizer.best_params['C_log'],
        'gamma': 10 ** svm_optimizer.best_params['gamma_log'],
        'kernel': svm_optimizer.best_params['kernel']
    }
    
    print("\nЛучшие параметры: ", best_params_transformed)
    
    print("Лучшее значение: ", svm_optimizer.best_value)

    # Сравниваем с моделью по умолчанию
    #svm_optimizer.compare_with_default()

    # Получаем лучшую модель
    #best_model = svm_optimizer.get_best_model()
    #print(f"\nЛучшая модель: {best_model}")

    # Дополнительно: можно сохранить лучшие параметры в файл
    #import json

    #best_params_transformed = {
    #    'C': 10 ** svm_optimizer.best_params['C_log'],
    #    'gamma': 10 ** svm_optimizer.best_params['gamma_log'],
    #    'kernel': svm_optimizer.best_params['kernel']
    #}

    #with open('best_svm_params.json', 'w') as f:
    #    json.dump(best_params_transformed, f, indent=4)
    #print("\nЛучшие параметры сохранены в best_svm_params.json")