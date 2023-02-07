import numpy as np
import cuquantum
import time
import json
import os
from explicit_XORSAT_TN_counting import (generate_initial_expr, 
                                         tensors_initialization,
                                         update_expr_and_tensors,
                                         count_theoretical_result)


class random_3Regular_XORSAT:
    """
    Purpose:
        Model a random 3-regular XORSAT problem as a tensor network and update step by step
        the tensors and their subscripts (einsum notation).

    Inputs:
        * number_of_variables (int): Number of variables in the XORSAT problem.
        * sample (int): sample chosen
    """
    def __init__(self, number_of_variables: int, sample: int):
        self.constraint_neighbours = []
        self.variable_neighbours = []
        self.sample = sample
        self.expr = ""
        self.operands = []
        self.number_of_variables = number_of_variables
        self.number_of_constraints = number_of_variables
        self.parity_vector = np.random.choice([0, 1], size = self.number_of_constraints)
    
    def get_data_from_json(self):
        """
        Collect data from a .json file.
        """
        graph = json.load(open(f"Data/3regularGraphs/N{self.number_of_variables}/{self.sample}.json"))
        self.constraint_neighbours = graph["graph"]["constraint_neighbors"]
        self.variable_neighbours = graph["graph"]["variable_neighbors"]

    def initialize_tensors(self):
        """
        Generate the initial list of tensors in the right order.
        """
        self.operands = tensors_initialization(self.number_of_variables, self.parity_vector)
        return self.operands

    def get_theoretical_result(self):
        """
        Evaluate the theoretical number of solutions for the given XORSAT problem.
        """
        theoretical_result = count_theoretical_result(self.constraint_neighbours, self.parity_vector)
        return theoretical_result

    def get_initial_expr(self):
        """
        Generate the initial expr for the tensors in the right order.
        """
        self.expr = generate_initial_expr(self.constraint_neighbours, self.variable_neighbours)
        return self.expr

    def update_info(self, path, new_ids):
        """
        Update the list of tensors and the expr after the one contraction step, given
        by cuquantum.Network.contract_path().
        """
        self.expr, self.operands, info = update_expr_and_tensors(path, self.operands, self.expr, new_ids)
        return self.expr, self.operands, info


os.system("clear")

def step_by_step_contraction(N, sample, show_subscripts=False):
    XORSAT_problem = random_3Regular_XORSAT(N, sample)
    XORSAT_problem.get_data_from_json()
    tensors_list = XORSAT_problem.initialize_tensors()
    theoretical_result = XORSAT_problem.get_theoretical_result()
    expr = XORSAT_problem.get_initial_expr()
    number_of_contractions = 2*N - 1

    if show_subscripts:
        print(f"expr at the start (step 0): {expr}")

    with cuquantum.Network(expr, *tensors_list) as tn:
        _, info = tn.contract_path({"samples": 500})

    for i in range(number_of_contractions):
        expr, tensors_list, info = XORSAT_problem.update_info(info.path, info.intermediate_modes[0])
        if show_subscripts:
            print(f"expr after contraction step {i+1}: {expr}")
        if len(expr.split(',')) == 2:
            print("------------------------------")
            with cuquantum.Network(expr, *tensors_list) as tn:
                _, info = tn.contract_path({'samples': 500}) # Why specify {'samples': 500} ?
                tn.autotune(iterations=5)
                result = tn.contract()
            break

    if theoretical_result == result:
        print(f"Theoretical result is: {theoretical_result}")
        print(f"Result after network contraction: {result}")
        print("  -> Step by step implementation succeeded!")
    else:
        print(f"Theoretical result is: {theoretical_result}")
        print(f"Result after network contraction: {result}")
        print("  -> Step by step implentation failed...")

if __name__ == "__main__":
    N = 8
    sample = 0
    step_by_step_contraction(N, sample, show_subscripts=True)
