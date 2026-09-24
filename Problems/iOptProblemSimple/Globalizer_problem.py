import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from typing import List
from rastrigin import Rastrigin
from trial import Point
from trial import FunctionValue
from problem import Problem

import requests
from tqdm import tqdm
import os
from pathlib import Path
import hashlib
import numpy as np
import shutil



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
    
    def finalize(self, coordinate: List[float], discreteCoordinate: List[str]):
        print("Finalize!!!")

def get_problem_parameters_names(class_name: str)->List[str]:
    if class_name == 'Rastrigin':
        return ["Dimension"]
    return []


def start_rastrigin():
    r_problem = Rastrigin(3)
    problem = GlobalizerProblem(r_problem)
    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([0.5] * problem.get_dimension(), [],0)
    sp = problem.get_start_value()
    print(result)
    resultAllFunction = problem.calculate_all_functionals([0.5] * problem.get_dimension(), [])
    print(resultAllFunction)


if __name__ == "__main__":
    start_rastrigin()