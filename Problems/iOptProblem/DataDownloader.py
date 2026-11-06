import argparse
from pathlib import Path
from typing import Optional
import os
import Loader as Loader

def find_project_folder(project_folder: str, max_level = 6):
    current = Path.cwd()
    i = 0
    while True:
        if i == max_level:
            break
        i += 1
        if current.name == project_folder and current.exists() and current.is_dir():
            return current
        parent = current.parent
        if parent == current:
            print("Найдена корневая папка!")
            break
        current = parent
    return None

def contains_build_folder(path: str):
    bin_path = Path(path)
    bin_path = bin_path.joinpath('_bin')
    if bin_path.exists():
        return bin_path
    return None

def parse_args(default_args = None):
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(
        description='Скрипт с выбором локализации загрузки датасета (global/local) и класса задачи',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --source global --class ECGClassification
  %(prog)s -s local -c ECGSegmentation
  %(prog)s --source global --class ECGGeneration --verbose
        """
    )

    # Добавляем флаг --source
    parser.add_argument(
        '-s', '--source',
        type=str,
        choices=['global', 'local'],
        required=True,
        default='local',
        help='Выберите источник: global или local'
    )

    # Добавляем флаг --class
    parser.add_argument(
        '-c', '--class',
        type=str,
        choices=['ECGClassification', 'ECGSegmentation', 'EEGClassification', 'FromAirObjectDetection', 'ECGPacemakerClassification'],
        required=True,
        dest='task_class',  # Используем dest чтобы избежать конфликта с keyword 'class'
        default='ECGClassification',
        help='Выберите класс задачи: ECGClassification, ECGSegmentation, ECGPacemakerClassification, FromAirObjectDetection...'
    )

    # Дополнительные опциональные аргументы
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод информации'
    )

    # Парсим аргументы
    if default_args:
        return parser.parse_args(default_args)
    else :
        return parser.parse_args()

task_links = {
    "ECGClassification" : ["https://disk.yandex.ru/d/2G-w_EXbKFb7QA", "ECG_Problems/Classification/dataset"],
    "ECGSegmentation" : ["https://disk.yandex.ru/d/M9xdGssonkuKgw", "ECG_Problems/Segmentation/dataset"],
    "FromAirObjectDetection": ["https://disk.yandex.ru/d/YLWsT_gKcT063w", "ObjectDetection/FromAirObjectDetection/dataset"],
    "ECGPacemakerClassification": ["https://disk.yandex.ru/d/UWHfil_TGYiO_w", "ECG_Problems/Pacemaker/dataset"],
    "EEGClassification": ["https://disk.yandex.ru/d/4tepqtF_RnC2Ig", "EEG_Problems/EEGClassification/dataset"]
}

if __name__ == '__main__':
    args = parse_args() #(['-s', 'global', '-c', 'ECGClassification', '-v'])

    path = None
    if args.source == 'local':
        path = Path(task_links[args.task_class][1])
    if args.source == 'global':
        path = contains_build_folder(find_project_folder("Globalizer"))
        if path:
            path = Path(os.path.join(path, 'dataset'))
            print(f"Папка найдена: {path}")
        else:
            raise FileNotFoundError("Папка не найдена! Возможно вы не собрали проект. Работа загрузчика остановлена ")

    # Создаём папку под датасеты если она отсутствовала
    if not path.exists():
        os.mkdir(path)

    #Создаём подпапку для датасета
    path = Path(os.path.join(path, args.task_class))
    if not path.exists():
        os.mkdir(path)

    Loader.Load(path, task_links[args.task_class][0])

