import numpy as np
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from trial import Point
from trial import FunctionValue
from trial import Trial
from problem import Problem
from Globalizer_problem import GlobalizerProblem


class Rastrigin(Problem):
    """
    The Rastrigin function is given by the formula:
       :math:`f(y)=(\sum_{i=1}^{N}[x_{i}^{2}-10*cos(2\pi x_{i})])`,
       where :math:`x\in [-2.2, 1.8], N` – problem dimensionality
    """

    def __init__(self, dimension: int):
        """
        The constructor of the Rastrigin problem class

        :param dimension: problem dimensionality.
        """
        super(Rastrigin, self).__init__()
        self.name = "Rastrigin"
        self.dimension = dimension
        self.number_of_float_variables = dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.dimension), dtype=str)
        for i in range(self.dimension):
            self.float_variable_names[i] = i

        self.lower_bound_of_float_variables = [-2.2] * self.dimension
        self.upper_bound_of_float_variables = [1.8] * self.dimension
        #self.lower_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        #self.lower_bound_of_float_variables.fill(-2.2)
        #self.upper_bound_of_float_variables = np.ndarray(shape=(self.dimension), dtype=np.double)
        #self.upper_bound_of_float_variables.fill(1.8)

        self.known_optimum = np.ndarray(shape=(1), dtype=Trial)

        pointfv = np.ndarray(shape=(self.dimension), dtype=np.double)
        pointfv.fill(0)
        KOpoint = Point(pointfv, [])
        KOfunV = np.ndarray(shape=(1), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0
        self.known_optimum[0] = Trial(KOpoint, KOfunV)

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of the selected function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated. 
        :param function_value: object defining the function number in the task and storing the function value.
        :return: Calculated value of the function at point.
        """
        sum: np.double = 0
        for i in range(self.dimension):
            sum += point.float_variables[i] * point.float_variables[i] - 10 * math.cos(
                2 * math.pi * point.float_variables[i]) + 10

        function_value.value = sum
        return function_value

    def get_name(self):
        return self.name


if __name__ == '__main__':
    r_problem = Rastrigin(3)
    problem = GlobalizerProblem(r_problem)
    bound = problem.get_lower_bounds()
    print(bound)
    result = problem.calculate([0.5] * problem.get_dimension(), [],0)
    sp = problem.get_start_value()
    print(result)
    resultAllFunction = problem.calculate_all_functionals([0.5] * problem.get_dimension(), [])
    print(resultAllFunction)