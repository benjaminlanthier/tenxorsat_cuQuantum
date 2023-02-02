import cupy as cp
import numpy as np
import cuquantum
from cuquantum import cutensornet as cutn
import json
import os
import signal
from pysat.formula import CNF
from pysat.solvers import Glucose4
import random 

"""
- Those lists show which boolean variable (x_i) is connected to which XOR constraint (c_i).

- This makes it possible ot find the way the graph should be connected.

- If, for example, we have this output:
    Constraints neighbours are :
    [[1, 0, 2], [1, 5, 4], [5, 1, 4], [6, 0, 7], [2, 4, 5], [6, 0, 3], [7, 6, 3], [2, 3, 7]]
    Variable neighbours are :
    [[5, 0, 3], [0, 1, 2], [4, 7, 0], [6, 7, 5], [4, 2, 1], [1, 2, 4], [5, 3, 6], [3, 6, 7]]
this means that c0 is connected to variables x1, x0 and x2, and c3 is connected to x6, x0 and x7.
This also means that the variable x0 is conected to c5, c0 and c3, and x3 is connected to c6, c7 and c5.

- All those indices need to be rewritten in order to be understood by the cuQuantum.Network() class.
This class uses similar indices notation as opt_einsum (as in it also accepts unicode characters, which
gives more indices possibilities).

- How to do this index rewriting in a smart way?
    * I could think of each of those variables/constraints as all simply tensors. In this case, the example
    shown above then becomes:
        --> Tensor t0 is connected to tensors t9, t8 and t10. So, x1 became t9, x0 became t8 and x2 became t10.
        --> Tensor t3 is connected to tensors t14, t8 and t11. So, x6 became t14, x0 became t8 and x7 became t15.
        --> Tensor t8 is connected to tensors t5, t0 and t3.
        --> Tensor t11 is connected to tensors t0, t1 and t2.
    In short, I need to "add 8" to the tensor's id only in the constraint_neighbours..?

    * I could define a list of unicode characters containing 3N characters. I then divide this list in groups of
    3 to define the indices of the constraint tensors immediately. I then need to give the right indices from
    those groups of 3 to the variable tensors. For doing so, I need to see which variable is connected to which
    constraints, look for the unicode character linked to the desired value and give this character as an index
    to the variable tensor. For doing so, I should mabe create a dictionnary containing the strings where the tensors
    are the keys and the indices are the values.
"""


def generate_initial_expr(constraint_neighbours, variable_neighbours):
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
    return initial_expr


# Not sure if this function is useful at the moment...
def generate_modes_in(constraint_neighbours, variable_neighbours):
    all_neighbours = [[num+len(constraint_neighbours) for num in array] for array in constraint_neighbours]
    for array in variable_neighbours:
        all_neighbours.append(array)
    # for i, array in enumerate(all_neighbours):
    #     print(f"{i}, {array}")
    # modes_in = tuple([num for num in array] for array in all_neighbours)
    modes_in_chr = tuple([chr(num+1) for num in array] for array in all_neighbours)
    num_modes_in = tuple(len(neighbours) for neighbours in all_neighbours)  
    return modes_in_chr, num_modes_in # first half of modes is for the constraint tensors and the other is for the variable tensors


# Not sure if this function is useful at the moment...
def generate_extents(nb_variables, dim=2, k=3):
    nb_tensors = 2 * nb_variables
    extents = tuple((dim,) * k for _ in range(nb_tensors))
    return extents


def xorTensor(parity, var_dim=2, k=3):
    dims = (var_dim,) * k
    tensor = cp.zeros(dims, dtype = float)
    for variable_ijk in range(var_dim**k):
        c = np.unravel_index(variable_ijk, dims) # Gives index of tensor
        if np.sum(c) % 2 == parity:
            tensor[c] = 1
    return tensor


def copyTensor(var_dim=2, k=3):
    dims = (var_dim,) * k
    tensor = cp.zeros(dims, dtype = float)
    tensor[cp.diag_indices(var_dim, k)] = 1
    return tensor


def tensors_initialization(nb_variables, parity_vector):
    variableTensor = copyTensor()
    tensors = []
    for parity in parity_vector:
        tensors.append(xorTensor(parity = parity))
    for _ in range(nb_variables):
        tensors.append(variableTensor)
    return tensors


def count_theoretical_result(clauses, parity_vector):
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
    solver = Glucose4(bootstrap_with=cnf.clauses) # Glucose3 seems to be faster than Solver (testing Glucose4)

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

    M = N # M is the number of constraints and in this case, it is equal to the number of variables
    parity_vector = np.random.choice([0, 1], size = M)
    # print(f"  * The parity vector is: p = {parity_vector}")
    # Generate the tensor network and contract it
    nb_variables = len(variable_neighbours)
    expr = generate_initial_expr(constraint_neighbours, variable_neighbours)
    # modes_in, num_modes_in = generate_modes_in(constraint_neighbours, variable_neighbours)
    tensors_list = tensors_initialization(nb_variables, parity_vector)
    with cuquantum.Network(expr, *tensors_list) as tn:
        path, info = tn.contract_path({'samples': 500})
        # print(info) 
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
    # n.free() # Use this line if the network is not used in a context (the 'with <Tensor Network> as tn:')
    # Evaluate theoretical result
    th_result = count_theoretical_result(constraint_neighbours, parity_vector)

    # Print results and if they match or not
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

time_limit = 240 # Set the time limit

signal.signal(signal.SIGALRM, timeout_handler) # Register the signal handler

signal.alarm(time_limit) # Set the alarm

try:
    if __name__ == "__main__": # I tested with N = 116 and it worked under 4 minutes on my laptop (Nvidia RTX3060 GPU)
        # Initialize data (N=8 and sample=0 outputs 2 solutions, N=8 and sample=4 outputs 1 solution)
        # N = 8
        # nb_samples = 1
        # samples = [4]
        # samples = np.arange(0, nb_samples) # Take the first nb_samples samples
        # samples = [random.randint(0, 99) for _ in range(nb_samples)] # Do tests with random samples
        show_neighbours = False # Show the arrays taken from the .json file
        show_GPU_info = False # Show the information of the GPU used

        # Define values for randomized tests
        nb_tests = 2
        N_min = 8
        N_max = 100
        path = "Data/3regularGraphs/"
        directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        numbers = [int(d.replace("N", "")) for d in directories if d.startswith("N")]

        # Clear the terminal to have a nice output
        os.system("clear")

        # Start the simulations
        # print(f"For N = {N}, we have:")
        for i in range(nb_tests):
            print(f"- Test {i}:")
            N = random.choice(numbers)
            sample = random.randint(0, 99)
            print(f"  * N = {N} with sample{sample}")
            main(N, sample,
                show_neighbours=show_neighbours,
                show_GPU_info=show_GPU_info)
except TimeoutException:
    print("Timeout expired...")
except Exception as error:
    print(error)
finally:
    signal.alarm(0) # Reset the alarm
