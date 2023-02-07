import cupy as cp
import numpy as np
import cuquantum
from cuquantum import cutensornet as cutn
import json
import os
import signal
from pysat.formula import CNF
from pysat.solvers import Glucose4



def generate_initial_expr(constraint_neighbours: list, variable_neighbours: list):
    """
    Purpose:
        Translate the variables and constraints neighbours notation to einsum format for the initial part.

    Inputs:
        * constraint_neighbours (list of arrays): Contains all the numerotation of the variable tensors that are connected to each constraints.
        * variable_neighbours (list of arrays): Contains all the numerotation of the constraint tensors that are connected to each variables.

    Output:
        * initial_expr (string): Indices of each tensors in the tensor network in the einsum format.
    """

    # Initializing the dictionaries for each cases
    neighbours_dict = {}
    ids_dict = {}
    variable_ids_dict = {}

    # Get all the notation clear and sorting the variables in the clauses to ease indices notation
    for key, array in enumerate(constraint_neighbours):
        neighbours_dict['c'+str(key)] = sorted(array)
    for key, array in enumerate(variable_neighbours):
        neighbours_dict['x'+str(key)] = sorted(array)

    # Define all the unicode characters necessary for this tensor network
    nb_edges = int(3*len(variable_neighbours)) # 3N is the total number of edges in our 3XORSAT tensor network
    start = 0x0061 #0x0100
    all_unicode_chr = ""
    end = start + nb_edges
    for unicode in range(start, end):
        all_unicode_chr += chr(unicode) # chr(...) gives the character of the unicode given as a string

    # Divide 'all_unicode_chr' in groups of 3, alphabetically, for the constraint tensors
    n = 3
    for constraint_number, i in enumerate(range(0, len(all_unicode_chr), n)):
        sub_str = all_unicode_chr[i:i+n]
        ids_dict['c'+str(constraint_number)] = sub_str

    # Associate the good indices to the variable tensors
    for var_number in range(len(variable_neighbours)):
        var_neighbours = neighbours_dict['x'+str(var_number)] # Find the array of constraints neighbours to this variable
        tensor_ids = ""
        for var_neighbour in var_neighbours:
            constr_neighbours = neighbours_dict['c'+str(var_neighbour)]
            position = constr_neighbours.index(var_number)
            var_index = ids_dict['c'+str(var_neighbour)][position]
            tensor_ids += var_index
        variable_ids_dict['x'+str(var_number)] = tensor_ids

    # Create one big dictionary with those indices llinked to their respective tensor
    ids_dict.update(variable_ids_dict)

    # Generate the expr necessary to use the cuquantum.Network class from the ids_dict dictionary
    initial_expr = ','.join(ids_dict.values())
    # print(f"Initial expr: {initial_expr}")
    return initial_expr


def update_expr_and_tensors(path: list, tensors_list: list, old_expr: str, new_ids: str):
    """
    Purpose:
        Update the 'expr' and the 'tensors_list' in the tensor network from
        the path evaluated with cuquantum.Network().contract_path.

    Inputs:
        * path (list of tuples): The optimized path given by cuquantum.Network.contract_path().
        * tensors_list (list of arrays): List of all the tensors represented
        in the tensor network.
        * old_expr (string): Previous 'expr' string given at the tensor network.
        * new_ids (string): Einsum ids for the new tensor.

    Outputs:
        * new_expr (string): The new 'expr' string after only one contraction
        in the network.
        * new_tensors_list (list of arrays): Updated tensors list after the
        contraction of 2 previous tensors.
        * new_info (Optimizer information): Information on the contraction of
        the tensor network.
    """

    # Get the ids of the tensors that will be contracted
    tensors_contracted = path[0] # First contraction done by the path
    t1id = tensors_contracted[0] # id number of the first tensor
    t2id = tensors_contracted[1] # id number of the second tensor

    # Update the old_expr_list to remove the contracted tensors ids and add the new one at the end
    old_expr_list = old_expr.split(',')
    str_id1_to_remove = old_expr_list[t1id]
    str_id2_to_remove = old_expr_list[t2id]
    old_expr_list.remove(str_id1_to_remove); old_expr_list.remove(str_id2_to_remove)
    old_expr_list.append(new_ids)
    new_expr = ','.join(old_expr_list)

    # Get the tensors that will be contracted and contract them to obtain the new tensor
    ti, tj = tensors_list[t1id], tensors_list[t2id]
    subscripts = ','.join([str_id1_to_remove, str_id2_to_remove]) + '->' + new_ids
    new_tensor = cuquantum.contract(subscripts, *[ti, tj])

    # Remove the contracted tensors from 'tensors_list' and add the new one at the end
    tensors_to_remove = [t1id, t2id]
    new_tensors_list = [tensor for i, tensor in enumerate(tensors_list) if i not in tensors_to_remove]
    new_tensors_list.append(new_tensor)

    # Find the new info from those updated inputs
    with cuquantum.Network(new_expr, *new_tensors_list) as tn:
        _, new_info = tn.contract_path() # Why specify {'samples': 500} ?

    return new_expr, new_tensors_list, new_info


def explicit_tn_contraction(initial_expr: str, initial_tensors_list: list, nb_sweeps=1):
    """
    Purpose:
        Explicitely contract each tensors following the optimized path given by
        cuquantum.Network.contract_path() so that we can do the sweeps (look for
        redundance) between each of them.

    Inputs:
        * initial_expr (string): Einsum indices notation of each tensors in the tensor network.
        * initial_tensors_list (list of arrays): List of all the tensors in the tensor network
        before of step of contraction and sweeps.
        * nb_Sweeps (int): Number of sweeps between the tensors contraction.

    Output:
        * result (int or float64): Number of solutions for the given 3XORSAT problem
        after the full contraction of the tensor network.
    """

    print(nb_sweeps)
    expr = initial_expr
    tensors_list = initial_tensors_list
    while len(tensors_list) >= 2:
        # Get the info out of the tensor network generated by cuQuantum
        with cuquantum.Network(expr, *tensors_list) as tn:
            _, info = tn.contract_path() # Why specify {'samples': 500} ?

        # Take the optimized path from info
        optimized_path = info.path

        # Update expr and tensors_list using optimized path's first step
        expr, tensors_list = update_expr_and_tensors(optimized_path, tensors_list)

        # Decompose (SVD or QR) the contracted tensor

        # Do the sweeps (which could completely modify the 'expr' and the 'tensors_list') PROBLEM...?

    """
        At this point, 'expr' and 'tensors_list' should be updated by a single contraction step from
        the given path. Now, what is left to do ?
        Steps to follow:
            1 - SVD/QR decompose this new tensor in order to see if there is redundance or not.
                * If there is not, go to the next step given by the new path evaluated after this
                    first contraction.
            2 - If we eliminated redundance after this decomposition, we need to propagate this info
                to the network, up to the point where it is not impacted by it anymore (do the sweeps).
            3 - Once the sweeps are over, modify the 'expr' and the 'tensors_list'.
            4 - Build the network that represents this updated version and re-do those steps until
                it is entirely contracted.
    """

    result = tensors_list[0]
    return result


def sweeps():
    # TO DO
    pass


def xorTensor(parity, var_dim=2, k=3):
    """
    Purpose:
        Generate the XOR constraint tensor.

    Inputs:
        * parity (bool): Boolean value that tells if the XOR contraint is satisfied when the
        binary sum of the variable in a clause is even (parity = 0) or odd (parity = 1).
        * var_dim (int): This means that the variables can take 'var_dim' values. In this case,
        'var_dim' = 2, which means that the variables are boolean.
        * k (int): Dimension of the constraint tensors. In this case, k = 3 since we work with
        the 3XORSAT problem, which means that we work with rank 3 tensors at the start.

    Output:
        * tensor (array): The XOR constraint tensor.
    """
    dims = (var_dim,) * k
    tensor = cp.zeros(dims, dtype = float)
    for variable_ijk in range(var_dim**k):
        c = np.unravel_index(variable_ijk, dims) # Gives index of tensor
        if np.sum(c) % 2 == parity:
            tensor[c] = 1
    return tensor


def copyTensor(var_dim=2, k=3):
    """
    Purpose:
        Generate the COPY variable tensor.

    Inputs:
        * var_dim (int): This means that the variables can take 'var_dim' values. In this case,
        var_dim = 2, which means that the variables are boolean.
        * k (int): Dimension of the variable tensors. In this case, k = 3 since we work with
        the 3XORSAT problem, which means that we work with rank 3 tensors at the start.

    Output:
        tensor (array): The COPY variable tensor.
    """
    dims = (var_dim,) * k
    tensor = cp.zeros(dims, dtype = float)
    tensor[cp.diag_indices(var_dim, k)] = 1
    return tensor


def tensors_initialization(nb_variables, parity_vector):
    """
    Purpose:
        Initialize a list of the constraint (XOR) and variable (COPY) tensors.

    Inputs:
        * nb_variables (int): Number of variables in the given 3XORSAT problem.
        * parity_vector (array): Array containing boolean values that tell if the XOR
        contraint is satisfied when the binary sum of the variable in a clause is even
        (parity = 0) or odd (parity = 1).

    Output:
        * tensors (list or arrays): List of the tensors, organized according the the
        generate_initial_expr() function, so [[xorTensors], [copyTensors]].
    """
    variableTensor = copyTensor()
    tensors = []
    for parity in parity_vector:
        tensors.append(xorTensor(parity = parity))
    for _ in range(nb_variables):
        tensors.append(variableTensor)
    return tensors


def get_tensors_from_network(tn):
    """
    Purpose:
        Extract tensors data out of the cuquantum Network class.

    Input:
        * tn (cuquantum.Network class): Tensor network generated by the Network
        class of cuquantum.

    Output:
        * readable_tensors_list (list of arrays): The current tensors in the network.
    """
    
    # tn.operands outputs a 'CupyTensor' object, unreadable directly
    readable_tensors_list = [cupyTensor_object.to() for cupyTensor_object in tn.operands]
    return readable_tensors_list

def count_theoretical_result(clauses, parity_vector):
    """
    Purpose:
        Evaluate the number of theoretical solutions of the given 3XORSAT problem.

    Inputs:
        * clauses (list of arrays): List of all the clauses that modelize the 3XORSAT problem.
        Form of each clause: (x_i XOR x_j XOR x_k)
        * parity_vector (array): Vector of boolean values (0s and 1s) where 0 means that the XOR
        constraint is satisfied when the sum of each variables in the clause is even and 1 means
        that the XOR constraint is satisfied when the sum of each variables in the clause is odd.

    Output:
        * count (int): Number of solutions that satisfy the given 3XORSAT problem.
    """

    # Initialize CNF format
    cnf = CNF()

    # Transform each XORSAT clause to 4 SAT clauses
    for clause, parity in zip(clauses, parity_vector):
        a, b, c = clause
        a+=1; b+=1; c+=1
        flip = int((-1)**parity)
        cnf.append([a, b, -flip *c])
        cnf.append([a, -b, flip * c])
        cnf.append([-a, b, flip * c])
        cnf.append([-a, -b, -flip * c])

    # Initialize solver
    solver = Glucose4(bootstrap_with=cnf.clauses) # Glucose3 seems to be faster than the general Solver (testing Glucose4)

    # Count theoretical number of solutions
    count = 0
    while solver.solve():
        count += 1
        solver.add_clause([-lit for lit in solver.get_model()])
    return count


def main(N, sample, show_neighbours = False, show_GPU_info = False):
    # Show GPU information if wanted
    if show_GPU_info:
        print("========================")
        print("cuTensorNet-vers:", cutn.get_version())
        dev = cp.cuda.Device()  # get current device
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        print("===== device info ======")
        print("GPU-name:", props["name"].decode())
        print("GPU-clock:", props["clockRate"])
        print("GPU-memoryClock:", props["memoryClockRate"])
        print("GPU-nSM:", props["multiProcessorCount"])
        print("GPU-major:", props["major"])
        print("GPU-minor:", props["minor"])
        print("========================")

    # Read data from .json file
    dataPath = "Data/"
    graph = json.load(open(dataPath + f"3regularGraphs/N{N}/{sample}.json"))
    constraint_neighbours = graph["graph"]["constraint_neighbors"]
    variable_neighbours = graph["graph"]["variable_neighbors"]

    # Show neighbours information if wanted
    if show_neighbours:
        print('Constraints neighbours are :')
        print(constraint_neighbours)
        print('Variable neighbours are :')
        print(variable_neighbours)

    # Generate a random parity vector p
    M = N # M is the number of constraints and in this case, it is equal to the number of variables
    parity_vector = np.random.choice([0, 1], size = M)

    # Generate the tensor network and contract it
    nb_variables = len(variable_neighbours)
    expr = generate_initial_expr(constraint_neighbours, variable_neighbours)
    # modes_in, num_modes_in = generate_modes_in(constraint_neighbours, variable_neighbours)
    tensors_list = tensors_initialization(nb_variables, parity_vector)
    with cuquantum.Network(expr, *tensors_list) as tn:
    # tn = cuquantum.Network(expr, *tensors_list)
        print(f"Some info about tn edges per vertex: {tn.inputs}") # tn.inputs
        # print(f"Some info about tn: {tn.inputs.}")
        _, info = tn.contract_path({'samples': 500})
        # path1, info1 = tn.qualifiers_in.all() # testing some functions...
        initial_path = info.path
        print('----------')
        print(info)
        print('----------')
        # print(tn.__getattribute__()))
        # for i in range(len(tn.operands)):
        #     tensor = tn.operands_data[i]
        #     ids = tn.allocator # tn.mode_map_ord_to_user.copy()
        #     print(f"Tensor {i} in the list: {tensor}") # cp.asnumpy(tensor)
        current_tensors_list = get_tensors_from_network(tn)
        current_subscripts = tn.inputs
        # new_list = cuquantum.cutensornet._internal.tensor_ifc_cupy.CupyTensor(cp.array([1,2,3,4,5])).to()
        for i in range(len(tn.operands)):
            print(f"Tensor {i}:\n {current_tensors_list[i]}, \nwith subscripts: {current_subscripts[i]}\n")
        print('----------')

        # for i, tree in enumerate(info.path):
        #     print(f"The contraction between tensors {tree[0]} and {tree[1]} gives a tensor with indices {info.intermediate_modes[i]}")
        '''
        info contains: 
            - largest intermediate,
            - optimized cost (in FLOPs),
            - the path,
            - slicing needed or not
            - the intermediate tensor mode labels
        '''
        tn.autotune(iterations=5)
        result = tn.contract()
    
    new_expr, new_tensors_list, new_Info = update_expr_and_tensors(info.path, tensors_list, expr, info.intermediate_modes[0])
    # tn.free() # Use this line if the network is not used in a context (the 'with <Tensor Network> as tn:')
    
    # Evaluate theoretical result
    th_result = count_theoretical_result(constraint_neighbours, parity_vector)

    # Print results and see if they match or not
    print(f"  * Result of the contraction is: {int(result)}")
    print(f"  * Theoretical result from pySAT is: {th_result}")
    if result == th_result:
        print(f"    -> Implementation succeeded!\n")
    else:
        print(f"    -> Implementation failed...\n")


#------------------------------Simulation starts here----------------------------------


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame): # the inputs are not used here, but they are necessary for it to work
    raise TimeoutException("Timeout expired")

time_limit = 240 # Set the time limit to 4 minutes

signal.signal(signal.SIGALRM, timeout_handler) # Register the signal handler

signal.alarm(time_limit) # Set the alarm

try:
    if __name__ == "__main__": # I tested with N = 116 and it worked under 4 minutes on my laptop (Nvidia RTX3060 GPU)
        # Initialize data (N=8 and sample=0 outputs 2 solutions, N=8 and sample=4 outputs 1 solution)
        N = 8
        samples = [0]
        show_neighbours = False # Show the arrays taken from the .json file
        show_GPU_info = False # Show the information of the GPU used

        # Clear the terminal to have a nice output
        os.system("clear")

        # Start the simulations
        # print(f"For N = {N}, we have:")
        for sample in samples:
            # print(f"  * N = {N} with sample{sample}")
            main(N, sample,
                show_neighbours=show_neighbours,
                show_GPU_info=show_GPU_info)
except TimeoutException:
    print("Timeout expired...")
except Exception as error:
    print(error)
finally:
    signal.alarm(0) # Reset the alarm
