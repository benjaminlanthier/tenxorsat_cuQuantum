"""
Purpose: Run test examples of the contraction to see the cost.

Date created: 2022-12-07
"""

import numpy as np
import quimb.tensor as qtn
# import cuquantum
# from cuquantum import cutensornet
from generators import biadjacencyGraph, xorsatFormula
from tensorFunctions import buildTensorNetwork, contract_explicit_tree
import cotengra
# import pycosat as pyco
from pysat.solvers import Solver
import random
import json
import os

if __name__ == "__main__":
    sweeps = 50
    cutoff = 1e-2 # For the SVD
    # Load samples:
    N = 8
    sample = 0
    dataPath = "Data/"
    graph = json.load(open(dataPath + f"3regularGraphs/N{N}/{sample}.json"))
    M = graph["number_of_constraints"]
    clauses = graph["graph"]["constraint_neighbors"]
    B = biadjacencyGraph(clauses, N)
    parity = np.array(random.choices([0,1], k = M))
    tensors = buildTensorNetwork(B, parity)
    for tensor in tensors:
        print(tensor)
    network = qtn.TensorNetwork(tensors)
    
    
    # Need to check if connected
    optimizer = cotengra.ReusableHyperOptimizer(methods = ["kahypar-agglom"], max_repeats = 16, max_time='equil:128', minimize = "size", reconf_opts={}, optlib = "nevergrad", progbar = True)
    #tree = network.contract(optimize = optimizer, get = "path-info")
    tree = network.contraction_tree(optimize = optimizer)
    print("Number of tensors: ", network.num_tensors)
    #contract, cost = contraction(network, tree, sweeps, cutoff)
    contract, cost = contract_explicit_tree(network, tree, sweeps, cutoff)
    #contractCompress = network.contract_compressed(optimize = optimizer, max_bond = None, cutoff = cutoff, callback_post_contract = compressByDistance, progbar = True)
    #print("Contract: ", contractCompress)
    tensorCount = 2**(N - network.select(tags = ["VARIABLE"]).num_tensors) * contract.contract()
    f = xorsatFormula(clauses, parity)
    satCount = np.sum([1 for solution in Solver(bootstrap_with = f).enum_models()])
    #satCount = np.sum([1 for solution in pyco.itersolve(f)])
    print("Contraction count: ", tensorCount)
    print("Actual count: ", satCount)
    print("Actual width: ", np.max(cost))
    print("Counts agree: ", np.allclose(satCount, tensorCount))
    #print("Theory width: ", int(tree.largest_intermediate))
    print("Full width: ", 2**tree.contraction_width())
