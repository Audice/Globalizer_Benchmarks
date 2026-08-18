import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from typing import List
import Cardio2D
from TestsProblem import TestsProblem
from rastrigin import Rastrigin
from trial import Point
from trial import FunctionValue
from problem import Problem
from MachineLearning.SupportVectorMachines import SVC_Fixed_Kernel
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
import requests
from tqdm import tqdm
import os
from pathlib import Path
import hashlib
import numpy as np
import shutil
from ECG.ECGClassificationProblem import ECGClassificationProblem
from MachineLearning.SupportVectorMachines.SVC_3D_Transformator import SVC_3D
from ECG.ECGSegmentationProblem import ECGSegmentationProblem
#from AirObjectDetectionProblem import AirObjectDetectionProblem
from ECG.PacemakerClassificationProblem import *
from EEG.EEGProblem import *


def _get_hash(path: Path) -> str:
    file_hash = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def download(path: Path, public_key: str) -> None:
    url = "https://cloud-api.yandex.net/v1/disk/public/resources"
    params = {"public_key": f"https://disk.yandex.ru/d/{public_key}"}

    response = requests.get(url, params=params).json()
    download_url = response["file"]
    file_size = response["size"]
    sha256 = response["sha256"]

    response = requests.get(download_url, stream=True)

    if path.is_file() and os.path.getsize(path) == file_size:
        print(f"File already downloaded: {path}")
        if _get_hash(path) == sha256:
            return

    with tqdm(total=file_size, unit="B", unit_scale=True) as progress_bar:
        with open(path, "wb") as f:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                f.write(data)

class GlobalizerProblem:

    def __init__(self, target_problem: Problem):
        self.dimension = target_problem.dimension
        self.problem: Problem = target_problem #Rastrigin(dimension=self.dimension)
        self.current_coordinate = List[float]
        self.point: Point = Point(float_variables=np.array(self.current_coordinate), discrete_variables=None)
        self.function_value: FunctionValue = FunctionValue()
        self.result : FunctionValue = FunctionValue()
        self.result_value = 0.0

        #self.set_dimension(self.dimension)

    def calculate(self, coordinate: List[float], discreteCoordinate: List[str], fNumber: int = 0) -> float:
        self.current_coordinate = coordinate
        self.point.float_variables = np.array(self.current_coordinate)
        self.point.discrete_variables = np.array(discreteCoordinate)
        self.function_value.functionID = fNumber
        self.result = self.problem.calculate(self.point, self.function_value)
        self.result_value = float(self.result.value)
        return self.result_value

    def calculate_all_functionals(self, coordinate: List[float], discreteCoordinate: List[str]) -> List[float]:
        self.current_coordinate = coordinate
        self.point.float_variables = np.array(self.current_coordinate)
        self.point.discrete_variables = np.array(discreteCoordinate)
        self.function_value.functionID = 0
        self.result = self.problem.calculate(self.point, self.function_value)
        self.result_value = float(self.result.value)
        return [self.result_value]

    def get_dimension(self) -> int:
        return self.dimension

    def get_number_of_functions(self) -> int:
        return self.get_number_of_criterions() + self.get_number_of_constraints()

    def get_number_of_constraints(self) -> int:
        return 0

    def get_number_of_criterions(self) -> int:
        return 1

    def get_start_y(self) -> List[float]:
        start_y = [(lb + ub) * 0.5 for lb, ub in
                   zip(self.problem.lower_bound_of_float_variables, self.problem.upper_bound_of_float_variables)]
        return start_y

    def get_start_value(self) -> List[float]:
        discrete_params = self.get_discrete_params()
        u = [i[0] for i in discrete_params]
        return [self.calculate(self.get_start_y(), u ,0)]

    def get_lower_bounds(self) -> List[float]:
        res = self.problem.lower_bound_of_float_variables
        return res

    def get_upper_bounds(self) -> List[float]:
        res = self.problem.upper_bound_of_float_variables
        return res

    def get_discrete_params(self) -> List[List[str]]:
        res = self.problem.discrete_variable_values
        for i in range(len(res)):
            res[i] = list(res[i])
        return res

    def set_dimension(self, dimension: int):
        pass
    #     self.dimension = dimension
    #     #self.problem: Problem = Rastrigin(dimension=dimension)
    #     self.current_coordinate = [0] * dimension
    #     self.point: Point = Point(float_variables=np.array(self.current_coordinate), discrete_variables=None)
    #     self.function_value: FunctionValue = FunctionValue()
    #     #self.result = self.problem.calculate(self.point, self.function_value)
    #     #self.result_value = float(self.result.value)

def get_problem_parameters_names(class_name: str)->List[str]:
    if class_name == 'Rastrigin':
        return ["Dimension"]
    if class_name == 'ECGClassificationProblem':
        return ["Dimension", "ProcRank"]
    if class_name == 'ECGSegmentationProblem':
        return ["Dimension", "ProcRank"]
    if class_name == 'EEGProblem':
        return ["Dimension", "ProcRank"]
    if class_name == 'TestsProblem':
        return ["DataSet", "Method"]
    if class_name == 'AirObjectDetectionProblem':
        return ["Dimension", "ProcRank"]
    if class_name == 'PacemakerClassificationProblem':
        return ["Dimension", "ProcRank"]
    return []



def load_breast_cancer_data():
    dataset = load_breast_cancer()
    x_raw, y_raw = dataset['data'], dataset['target']
    inputs, outputs = shuffle(x_raw, y_raw ^ 1, random_state=42)
    return inputs, outputs

def test_ecg_classification_problem():
    problem_ecg_class = ECGClassificationProblem(2)

    problem = GlobalizerProblem(problem_ecg_class)

    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([0.5, 100], [], 0)
    print(result)
    result_all_function = problem.calculate_all_functionals([0.5, 100], [])
    print(result_all_function)

def test_svc1d_problem():
    x, y = load_breast_cancer_data()
    kernel_coefficient = -5
    regularization_value_bound = {'low': 1, 'up': 6}
    problem_svc = SVC_Fixed_Kernel.SVC_Fixed_Kernel(x, y, kernel_coefficient, regularization_value_bound)

    problem = GlobalizerProblem(problem_svc)

    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([2], [], 0)
    print(result)
    result_all_function = problem.calculate_all_functionals([2], [])
    print(result_all_function)

def test_segmentation_problem():
    if not os.path.exists('data'):
        path = Path('data.zip')
        download(path, 'Oqxcid6uX58kYQ')
        shutil.unpack_archive('data.zip', 'data', format="zip")
        os.remove('data.zip')

    p_value_bound = {'low': 0.0, 'up': 1.0}
    q_value_bound = {'low': 1.0, 'up': 1.6}
    problem_nn = Cardio2D.Cardio2D(p_value_bound, q_value_bound)

    problem = GlobalizerProblem(problem_nn)

    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([0.5, 1.3], [], 0)
    print(result)
    result_all_function = problem.calculate_all_functionals([0.5, 1.3], [])
    print(result_all_function)

def test_rastrigin():
    r_problem = Rastrigin(3)
    problem = GlobalizerProblem(r_problem)
    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([0.5] * problem.get_dimension(), [],0)
    sp = problem.get_start_value()
    print(result)
    resultAllFunction = problem.calculate_all_functionals([0.5] * problem.get_dimension(), [])
    print(resultAllFunction)

def TestsProblemTest():
    problem_ecg_class = TestsProblem('balance', 'svc')

    problem = GlobalizerProblem(problem_ecg_class)
    des = problem.get_discrete_params()
    sp = problem.get_start_value()


    result = problem.calculate([1e-7, 1e2], ['rbf'])
    print(result)

def TestSVC3D():
    problem_transformator = SVC_3D()

    problem = GlobalizerProblem(problem_transformator)

    des = problem.get_discrete_params()
    sp = problem.get_start_value()

    #result = problem.calculate([6.0, -2.0], ['rbf'])
    #print("transformator SVC 3D in (6.0, -2.0, rbf) = " + str(result))
    default_result = problem_transformator.default_calculate()
    print("transformator SVC 3D default value = " + str(default_result))


def test_ecg_segmentation_problem_main():
    problem_ecg_class = ECGSegmentationProblem(5,0)

    problem = GlobalizerProblem(problem_ecg_class)

    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([4, 2, 0.3, 0.0005],['relu'])
    print(result)

def test_eegproblem():
    eegProblem = EEGProblem()

    problem = GlobalizerProblem(eegProblem)

    result = problem.calculate([0.75, 64, 1e-2, 512],['16', 'simple'])
    print(result)

def test_pacemaker_problem():
    pacemakerProblem = PacemakerClassificationProblem()

    problem = GlobalizerProblem(pacemakerProblem)

    result = problem.calculate([2, 16, 0.2, 0.0001, 0.00005, 0.5, 1.0, 3], [])
    print(result)

#def test_air_object_detected_problem():
#    problem_class = AirObjectDetectionProblem(2)
#    problem = GlobalizerProblem(problem_class)
#    bound = problem.get_lower_bounds()
#    print(bound)
#    result = problem.calculate([-3.0], ["32", "5"])
#    print(result)

if __name__ == "__main__":
    test_pacemaker_problem()
    #test_eegproblem()
    #test_air_object_detected_problem()
    #test_ecg_segmentation_problem_main()    
    #TestSVC3D()
    #TestsProblemTest()
    #test_ecg_classification_problem()
    #test_svc1d_problem()
    #test_segmentation_problem()
    #test_rastrigin()