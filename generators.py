"""
Purpose: Functions to generate the XORSAT problems we will contract.

Date created: 2022-12-07
"""

import numpy as np
# import cupy as cp
import random


def generateClauses(N, M, k = 3):
    """
        - Purpose: Generate clauses by randomly choosing k of the N variables for each clause.
        - Inputs:
            - N (integer): The number of variables to generate.
            - M (clauses): The total number of clauses to generate.
            - k (integer): The number of variables per constraint.
        - Outputs:
            - clausePositions (list): The list of positions where the clauses are being applied.
    """

    clausePositions = []

    while len(clausePositions) < M:
        pos = random.sample(range(N), k = k)
        pos.sort()
        if pos not in clausePositions:
            clausePositions.append(pos)

    return clausePositions

def biadjacencyGraph(clauses, N):
    """
        - Purpose: Transform the triples to a biadjacency matrix.
        - Inputs:
            - triples (list): The triples which form the clauses.
            - N (integer): The number of variables in the instance.
        - Outputs:
            - clauses (array): The biadjacency matrix.
    """
    M = len(clauses)
    biadjacency = np.zeros((M,N), dtype = int)

    r = 0 # Row index
    for clause in clauses:
        biadjacency[r, clause] = 1
        r += 1

    return biadjacency

def xorsatFormula(clauses, parity):
    """
        - Purpose: Create the CNF formula for a SAT solver.
                Note: For now, it just works with 3-XORSAT.
        - Inputs:
            - clauses (list of lists): The clauses for the XORSAT problem.
            - parity (array): The parity for the linear equations.
        - Output:
            - formula (list): The list of constraints.
    """
    formula = []
    for clause, y in zip(np.array(clauses), parity):
        flip = np.ones(len(clause))
        flip[0] = (-1)**y
        for k in range(2**(len(clause)-1)):
            mask = np.ones(len(clause))
            if k < 2**(len(clause)-1) - 1:
                mask[k] = -1
            else:
                mask *= -1
            result = mask * flip * (clause + 1) # +1 because SAT solvers start with index 1
            formula.append([int(i) for i in result])
    return formula

def satFormula(clauses, polarity):
    """
        - Purpose: Create the CNF formula for a SAT solver.
                Note: For now, it just works with 3-XORSAT.
        - Inputs:
            - clauses (list of lists): The clauses for the SAT problem.
            - polarity (array): The polarities of the SAT constraints as integers.
        - Output:
            - formula (list): The list of constraints.
    """
    formula = []
    for clause, y in zip(np.array(clauses), polarity):
        y = np.binary_repr(y)
        y = [int(b) for b in y]
        y = (-1)**np.array(y)
        result = y * (clause + 1) # +1 because SAT solvers start with index 1
        formula.append([int(i) for i in result])
    return formula