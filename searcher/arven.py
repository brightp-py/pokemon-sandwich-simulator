import numpy as np
import json
from collections import deque
from scipy.optimize import linprog

import data

def intSimplex(c, A, b):
    
    queue = deque([(A, b)])

    while queue:
        A0, b0 = queue.popleft()
        x = linprog(c, A_ub=A0, b_ub=b0, bounds=(None, 0),
                    method='highs-ds').x
        if x is None:
            continue
        x = -x
        # print(x)
        
        dec = x % 1
        if np.all(dec == 0):
            yield x
            continue
        pivot_ind = np.argmax(dec)

        bottom_row = np.zeros(A0.shape[1])
        bottom_row[pivot_ind] = 1

        Al = np.concatenate((A0, -bottom_row[np.newaxis, :]))
        Ar = np.concatenate((A0, bottom_row[np.newaxis, :]))

        bl = np.concatenate((b0, np.array([np.floor(x[pivot_ind])])))
        br = np.concatenate((b0, np.array([-np.ceil(x[pivot_ind])])))

        queue.append((Al, bl))
        queue.append((Ar, br))

def buildQuery(flavors, powers, types):
    assert len(flavors) == 2 and len(set(flavors)) == 2
    assert len(powers) == 3 and len(set(powers)) == 3
    assert len(types) == 3

    A = data.MATRIX.copy()
    b = np.zeros(A.shape[1])

    """ RULE 0
    We do not yet care what the actual values for Flavor 1, Power 1, and Type 1
    are. We only care about how they relate to the others. But we will need
    this rule when we take Power levels into account.
    Formally:
        0 <= 0
    """
    flav1 = data.row2ind[flavors[0]]
    power1 = data.row2ind[powers[0]]
    ele1 = data.row2ind[types[0]]
    A[flav1] = 0
    A[power1] = 0
    A[ele1] = 0
    
    """ RULE 1
    Flavor 1 >  Flavor 2  if Flavor 2 comes before Flavor 1
    Flavor 1 >= Flavor 2  otherwise
    Formally:
        Flav2 - Flav1 <= -1, if ind(Flav2) < ind(Flav1)
        Flav2 - Flav1 <=  0, if ind(Flav2) >= ind(Flav1)
    """
    flav2 = data.row2ind[flavors[1]]
    A[flav2] -= data.MATRIX[flav1]
    if flav2 < flav1:
        b[flav2] = -1

    """ Rule 2
    Flavor 2 > Flavor i  for all 0 <= i < 5 and i != Flavor 1, Flavor 2
    Formally:
        FlavI - Flav2 <= -1, if ind(FlavI) < ind(Flav2)
        FlavI - Flav2 <=  0, if ind(FlavI) >= ind(Flav2)
    """
    flavI = list(range(5))
    flavI.remove(flav1)
    flavI.remove(flav2)
    flavI = np.array(flavI)
    A[flavI] -= data.MATRIX[flav2]
    b[flavI[flavI < flav2]] = -1

    print(A[:5])

buildQuery(
    ['Sweet', 'Sour'],
    ['Catching', 'Item Drop', 'Raid'],
    ['Flying', 'Steel', 'Ice']
)
