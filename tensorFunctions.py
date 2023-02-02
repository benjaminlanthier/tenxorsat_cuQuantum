"""
Purpose: Functions to build the tensor network given data. Some of these functions
         are based off of tensorCSP from Stefanos Kourtis

Date created: 2022-12-07
"""

import numpy as np
# import cupy as cp
import quimb.tensor as qtn
import time
import cotengra

def xorGate(i, m):
    """
        - Purpose: Computes the XOR of the integers i and m (as bitstrings),
                   then sums them up mod 2.

                   Note: I use a NOT when returning the function because the XOR
                         is valid when the result is even.
        - Inputs:
            - i (int): The configuration of variables.
            - m (int): The polarity vector of the variables.
        - Output:
            - x (Boolean): The result of the XOR.
    """
    x = np.binary_repr(i ^ m)
    x = [int(b) for b in x]
    #return ~((np.sum(x) % 2) > 0)
    return np.sum(x) % 2 == 0

def orGate(i, m):
    """
        - Purpose: Compute OR of inputs.
        - Inputs:
            - i (int): The configuration of variables.
            - m (int): The polarity vector of the variables.
        - Output:
            - Boolean: The result of the OR.
    """
    x = np.binary_repr(i ^ m)
    x = [int(b) for b in x]
    return np.sum(x) > 0

def satTensor(k, m = 0, gate = xorGate, q = 2):
    """
        - Purpose: Create the sat tensor.
                Note: This isn't efficient for arbitrary k!
                      I'm going through all possibilities.
        - Inputs:
            - k (int): The degree of the tensor.
            - m (int): The polarity of the constraint. Think of
                       the int as a binary vector. 
            - q (int): The dimension of the domain.
        - Outputs:
            - sat (array): The tensor.
    """
    d = [q] * k
    sat = np.zeros(d, dtype = float)
    for bits in range(q**k):
        c = np.unravel_index(bits, d) # Gives index of tensor
        sat[c] = gate(bits, m)
    
    return sat

def xorTensor(k, y, q = 2):
    """
        - Purpose: Create the XOR tensor.
                Note: This isn't efficient for arbitrary k!
                      I'm going through all possibilities.
        - Inputs:
            - k (int): The degree of the tensor.
            - y (int): The parity of the constraint.
            - q (int): The dimension of the domain.
        - Outputs:
            - xor (array): The tensor.
    """
    d = [q] * k
    xor = np.zeros(d, dtype = float)
    for bits in range(q**k):
        c = np.unravel_index(bits, d) # Gives index of tensor
        if np.sum(c) % 2 == y:
            xor[c] = 1
    
    return xor

def copyTensor(k, q = 2, dtype = float):
    """
        - Purpose: Create the copy tensor which holds variable indices.
        - Inputs:
            - k (int): The degree of the tensor.
            - q (int): Dimension of domain (2 is Boolean).
        - Outputs:
            - copy (array): The tensor array.
    """
    copy = np.zeros([2]*k, dtype = dtype)
    copy[np.diag_indices(q,k)] = True
    return copy

def buildTensorNetwork(B, parity, gate = xorGate):
    """
        - Purpose: Build the tensor network associated to the SAT problem.
        - Inputs:
            - B (array of shape (M,N)): The biadjacency matrix for the constraints.
            - parity (array of shape (M,)): The polarity for the constraints.
            - gate (function): The type of SAT gate.
        - Outputs:
            - tensors (list of tensors): The quimb tensor network object. 
    """
    tensors = []
    variableDictionary = {}
    # Loop for constraint tensors
    for c, (constraint, y) in enumerate(zip(B, parity)):
        variables = np.nonzero(constraint)[0]
        constraintIndices = []
        for variable in variables:
            index = "c"+str(c) + "x" + str(variable)
            constraintIndices.append(index)
            try:
                variableIndices = variableDictionary[variable]
                variableIndices.append(index)
            except:
                variableIndices = [index]
            variableDictionary[variable] = variableIndices
        #data = xorTensor(k = len(variables), y = y)
        data = satTensor(k = len(variables), m = y, gate = gate)
        constraint = qtn.Tensor(data = data, inds = constraintIndices, tags = ["C"+str(c), "CONSTRAINT"])
        tensors.append(constraint)

    # Loop for variable tensors
    for variable, indices in variableDictionary.items():
        data = copyTensor(k = len(indices))
        variableTensor = qtn.Tensor(data = data, inds = indices, tags = ["X"+str(variable), "VARIABLE"])
        tensors.append(variableTensor)
    return tensors


def bondDistances(network, root):
    """
        - Purpose: Given a root node in the tensor network, output
                a dictionary of indices to contract sorted by
                distance to the root.

                Note: If the graph is disconnected, then the
                function only returns distances < infinity.
        - Inputs:
            - network (TensorNetwork)
            - root (tid): The tensor ID.
        - Output:
            - distances (dictionary): The sorted dictionary of edges.
    """
    distances = {}
    for edge, nodes in network.ind_map.items():
        i, j = nodes
        if i == root or j == root:
            distances[edge] = 0
        else:
            d1 = network.compute_shortest_distances(tids = [i, root])
            d2 = network.compute_shortest_distances(tids = [j, root])
            if len(d1) != 0:
                distances[edge] = min(d1[tuple(sorted((i, root)))], d2[tuple(sorted((j, root)))])
    return dict(sorted(distances.items(), key=lambda x:x[1]))


def verify_all_zeros(network, cutoff=1e-4):
    """
        - Purpose: Verify if a tensor in the network contains only zeros.
        - Inputs:
            - network (quimb object): The current tensor network.
        - Outputs:
            - boolean value: *True means that the tensor network contains at least
                              one tensor that contains only zeros (or close to 0).
                             *False means that the tensor network does not contain
                              a tensor that contains only zeros (or close to 0).
    """

    tensors = network.tensors
    sizes = [tensor.shape for tensor in tensors]
    for i, tensor in enumerate(tensors):
        zeros_tensor = np.zeros(sizes[i], dtype=float)
        if np.allclose(zeros_tensor, tensor.data, atol=cutoff):
            return True
    return False


def compressByDistance(network, root, inplace=False, **compress_opts):
    """
        - Purpose: Starting from a root node, compress the bonds
                   in the tensor network.
        - Inputs:
            - network (quimb object): The tensor network.
            - root (tid): The tensor ID.
        - Outputs:
            - tn (quimb object): The new tensor network.
            - zeros_found (boolean): Tells if an all-zero tensor has been found in
                                     the tensor network.
    """
    tn = network if inplace else network.copy()
    tn.fuse_multibonds_()
    zeros_found = False

    distances = bondDistances(tn, root)
    for ix in tuple(distances):
        try:
            tid1, tid2 = tn.ind_map[ix]
        except (ValueError, KeyError):
            # not a bond, or index already compressed away
            continue
        tn._compress_between_tids(tid1, tid2, **compress_opts)

    if verify_all_zeros(tn):
        zeros = np.zeros((2,))
        quimb_zeros1 = qtn.Tensor(zeros, inds=['a'])
        zeros_tensors = [quimb_zeros1, quimb_zeros1]
        tn = qtn.TensorNetwork(zeros_tensors)
        print("At least one all-zero tensor has been found in the network (after first sweep step).")
        zeros_found = True
        return tn, zeros_found

    # Reverse
    for ix in reversed(tuple(distances)):
        try:
            tid1, tid2 = tn.ind_map[ix]
        except (ValueError, KeyError):
            # not a bond, or index already compressed away
            continue
        tn._compress_between_tids(tid1, tid2, **compress_opts)

    if verify_all_zeros(tn):
        zeros = np.zeros((2,))
        quimb_zeros1 = qtn.Tensor(zeros, inds=['a'])
        zeros_tensors = [quimb_zeros1, quimb_zeros1]
        tn = qtn.TensorNetwork(zeros_tensors)
        print("At least one all-zero tensor has been found in the network (after last sweep step).")
        zeros_found = True
        return tn, zeros_found
    
    return tn, zeros_found


def contraction(network, tree, sweeps = 1, cutoff = 1e-3):
    """
        - Purpose: Given a network and a contraction tree,
                   contract the network while performing sweeps
                   to simplify the size of the network.
        - Inputs:
            - network (quimb object): The tensor network.
            - tree (quimb or cotengra object): The contraction path.
            - sweeps (integer): Number of times to do the simplification
                                procedure.
            - cutoff (float): The truncation value for SVDs.
        - Outputs:
            - current (quimb object): The contracted tensor network.
            - cost (list): The size of the tensors during each step.
    """

    map = tree.quimb_symbol_map
    current = network.copy()
    current = current.astype(float)
    tensors = current.tensors
    sizes = [np.prod(t.shape) for t in tensors]
    #print("Initi: ", sizes)
    contractionList = tree.contraction_list
    #order = np.argsort(tree.scale_list)
    cost = []
    i = 0
    for t in contractionList:
    #for index in order:
        #t = contractionList[index]
        print("Contraction: ", i / len(contractionList))
        i += 1
        #s = list(t[1])[0] # This grabs only one contraction symbol, but I associate it to the variable tags so that it contracts potentially multiple indices at once.
        symbols = list(t[1])
        #print(t)
        #print("Symbols: ", symbols)
        start = time.time()
        for x in symbols:
            edge = map[x]
            #print("x: ", x)
            edge = map[x]
            #print("Edge: ", edge)
            tag = edge.split("x")
            variableTag = "X" + tag[1]
            constraintTag = tag[0]
            constraintTag = "C" + constraintTag.split("c")[1]
            current.contract_between(tags1 = variableTag, tags2 = constraintTag)
            #current.contract_ind(x)
            comparisonSizes = np.array([np.prod(t.shape) for t in current.tensors])
            #print("initi: ", comparisonSizes)
            contractTime = time.time()

        #print("Tensors: ")
        #print(current.tensors)
        #print([t.data for t in current.tensors])
        #print("Tag map: ", current.tag_map)
        root = list(current.tag_map[variableTag])[0]
        #print("Root: ", root)
        comparisonSizes = np.array([np.prod(t.shape) for t in current.tensors])
        #cost.append(np.sum(comparisonSizes)) # Add all tensors together
        cost.append(np.max(comparisonSizes)) # Just the largest tensor
        print("initi: ", comparisonSizes)
        for _ in range(sweeps):
            current, _ = compressByDistance(current, root, cutoff = cutoff)
            current = current.compress_all()
            compressTime = time.time()
            tensors = current.tensors
            sizes = np.array([np.prod(t.shape) for t in tensors])
            if not np.allclose(comparisonSizes, sizes):
                print("Sizes: ", sizes)
                #print("Simplified")
                comparisonSizes = np.copy(sizes)
            else:
                break
        #print("Contraction: ", i / len(contractionList))
        print("Contract time: ", contractTime - start)
        print("Compress time: ", compressTime - contractTime)
        print("Cost: ", cost[-1])
        print()
    return current, cost

def contract_explicit_tree(network, tree, sweeps = 1, cutoff = 1e-5, output_inds=None, inplace=False):
    tn = network if inplace else network.copy()
    tn = tn.astype(float)
    # map between nodes in the tree and tensor ids in the network
    tidmap = dict(zip(tree.gen_leaves(), tn.tensor_map.keys()))

    numberOfContractions = tn.num_tensors - 1
    cost = []
    i = 0
    for parent, left, right in tree.traverse():
        #print("Contraction: ", i / (numberOfContractions))
        i += 1
        # get tensor ids
        tidl = tidmap.pop(left)
        tidr = tidmap.pop(right)

        # the following is like tn._contract_between_tids(tidl, tidr)

        # compute the new indices (only needed if hyper tensor network)
        new_inds = tn.compute_contracted_inds(
            tidl, tidr, 
            # output_inds here are those of the global contraction
            output_inds=output_inds
        )

        # pop the left and right tensors
        tl = tn._pop_tensor(tidl)
        tr = tn._pop_tensor(tidr)

        start = time.time()
        # do the contraction! (n.b. you could just do `tl @ tr` if the 
        # tensor network only has 'standard' indices)
        tp = qtn.tensor_contract(
            tl, tr, 
            # output_inds here are those of the local contraction
            output_inds=new_inds,
            # always wrap as a Tensor, even if scalar 
            preserve_tensor=True,
        )
        contractTime = time.time()

        # add tensor back specifically with tidr
        tn.add_tensor(tp, tid=tidr)

        # update the tensor map
        tidmap[parent] = tidr

        # Trim values that are near zero
        """
        for t in tn.tensors:
            data = t.data
            minm = np.abs(data).max() * cutoff  # minimum value tolerated
            data[np.abs(data) < minm] = 0
            t.modify(data = data)
        """
        

        #tn.column_reduce(atol = cutoff)
        #print("Tensors: ")
        #print(tn.tensors)
        #print([t.data for t in tn.tensors])
        root = tidr
        #print("Root: ", root)
        comparisonSizes = np.array([t.size for t in tn.tensors])
        #cost.append(np.sum(comparisonSizes)) # Add all tensors together
        cost.append(np.max(comparisonSizes)) # Just the largest tensor
        #print("initi: ", comparisonSizes)
        for _ in range(sweeps):
            tn, zeros_found = compressByDistance(tn, root, inplace = False, cutoff = cutoff)
            #tn.compress_all(inplace = True, cutoff = cutoff)
            compressTime = time.time()
            tensors = tn.tensors
            sizes = np.array([t.size for t in tensors])
            if zeros_found:
                return tn, cost
            if not np.allclose(comparisonSizes, sizes):
                #print("Sizes: ", sizes)
                #print("Simplified")
                comparisonSizes = np.copy(sizes)
            else:
                break
        #tn.rank_simplify(inplace = True)
        
        #print("Contract time: ", contractTime - start)
        #print("Compress time: ", compressTime - contractTime)
        #print("Cost: ", cost[-1])
        #print()
    return tn, cost