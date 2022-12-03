import numpy as np
import json
from collections import deque
from scipy.optimize import linprog

import data

def intSimplex(A, b, npieces):
    
    if isinstance(A, list) and isinstance(b, list):
        queue = deque(zip(A, b))
        c = np.ones(A[0].shape[1])
    else:
        queue = deque([(A, b)])
        c = np.ones(A.shape[1])

    while queue:
        A0, b0 = queue.popleft()
        x = linprog(c, A_ub=A0, b_ub=b0, method='highs-ds').x
        if x is None:
            continue
        # x = -x
        # print(x)
        
        # dec = (x * npieces) % 1
        dec = x % 1
        if np.all(dec == 0):
            yield x
            continue
        pivot_ind = np.argmax(dec)

        bottom_row = np.zeros(A0.shape[1])
        bottom_row[pivot_ind] = 1
        # bottom_row = bottom_row[np.nonzero(x)]
        # A0 = np.delete(A0, x == 0, axis=1)

        Al = np.concatenate((A0, bottom_row[np.newaxis, :]))
        Ar = np.concatenate((A0, -bottom_row[np.newaxis, :]))

        # down = np.floor(x[pivot_ind] * npieces[pivot_ind]) / npieces[pivot_ind]
        # up = np.ceil(x[pivot_ind] * npieces[pivot_ind]) / npieces[pivot_ind]
        down = np.floor(x[pivot_ind])
        up = np.ceil(x[pivot_ind])

        bl = np.concatenate((b0, np.array([down])))
        br = np.concatenate((b0, np.array([-up])))
        # bd = np.concatenate((b0, np.array([0])))

        queue.append((Al, bl))
        queue.append((Ar, br))
        # queue.append((Al.copy(), bd))

def determineSandwich(vec):
    A = np.sum(data.MATRIX * vec, axis=1)[:5]
    flav1 = np.argmax(A)
    A[flav1] = 0
    flav2 = np.argmax(A)
    flav1, flav2 = data.ind2row[flav1], data.ind2row[flav2]
    bonus = data.FLAVOR_BONUS[flav1][flav2]
    return f"{flav1}-{flav2} {bonus} Sandwich"

def printRecipe(vec):
    print()
    print(determineSandwich(vec))
    for ind, count in enumerate(vec):
        if count > 0:
            name = data.ind2col[ind]
            print(f" - {name} x {str(count)}")

def buildExactQuery(flavors, powers, types, lv2):
    assert len(flavors) == 2 and len(set(flavors)) == 2
    assert len(powers) == 3 and len(set(powers)) == 3
    assert len(types) == 3

    A = data.MATRIX.copy()
    b = np.zeros(A.shape[0])
    # c = -np.ones(A.shape[1])

    """ RULE 0
    We do not care what the actual value for Flavor 1 is.
    But for Powers, they should be greater than or equal to 100 if their level
    is 2, and for Types, the same logic applies, but >= 180.
    Formally:
        0 <= 0,          if #level2 = 0
         Power1 <=   99, if #level2 = 0
        -Power1 <= -100, if #level2 = 1
        -Power2 <= -100, if #level2 = 2
        -Power3 <= -100, if #level2 = 3
    """
    flav1 = data.row2ind[flavors[0]]
    A[flav1] = 0

    power1 = data.row2ind[powers[0]]
    power2 = data.row2ind[powers[1]]
    power3 = data.row2ind[powers[2]]
    ele1 = data.row2ind[types[0]]
    ele2 = data.row2ind[types[1]]
    ele3 = data.row2ind[types[2]]
    A[power1] = 0
    A[ele1] = 0

    if lv2 != 0:
        if lv2 == 1:
            A[power1] = -data.MATRIX[power1]
            A[ele1] = -data.MATRIX[ele1]
        elif lv2 == 2:
            A[power1] = -data.MATRIX[power2]
            A[ele1] = -data.MATRIX[ele2]
        elif lv2 == 3:
            A[power1] = -data.MATRIX[power3]
            A[ele1] = -data.MATRIX[ele3]
        b[power1] = -100
        b[ele1] = -180

    
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
    
    """ RULE 3
    Power 1 >  Power 2  if Power 2 comes before Power 1
    Power 1 >= Power 2  otherwise
    Formally:
        Power2 - Power1 <= -1, if ind(Power2) < ind(Power1)
        Power2 - Power1 <=  0, if ind(Power2) >= ind(Power1)
    """
    A[power2] -= data.MATRIX[power1]
    if power2 < power1:
        b[power2] = -1
    
    """ RULE 4
    Power 2 >  Power 3  if Power 3 comes before Power 2
    Power 2 >= Power 3  otherwise
    Formally:
        Power3 - Power2 <= -1, if ind(Power3) < ind(Power2)
        Power3 - Power2 <=  0, if ind(Power3) >= ind(Power2)
    """
    A[power3] -= data.MATRIX[power2]
    if power3 < power2:
        b[power3] = -1

    """ Rule 5
    Power 2 > Power i  for all 5 <= i < 15 and i != Power 1, Power 2, Power 3
    Formally:
        PowerI - Power2 <= -1, if ind(PowerI) < ind(Power2)
        PowerI - Power2 <=  0, if ind(PowerI) >= ind(Power2)
    """
    powerI = list(range(5, 15))
    powerI.remove(power1)
    powerI.remove(power2)
    powerI.remove(power3)
    powerI = np.array(powerI)
    A[powerI] -= data.MATRIX[power3]
    b[powerI[powerI < power3]] = -1
    
    """ RULE 6
    Type 1 >  Type 2  if Type 2 comes before Type 1
    Type 1 >= Type 2  otherwise
    Formally:
        Type2 - Type1 <= -1, if ind(Type2) < ind(Type1)
        Type2 - Type1 <=  0, if ind(Type2) >= ind(Type1)
    """
    A[ele2] -= data.MATRIX[ele1]
    if ele2 < ele1:
        b[ele2] = -1
    
    # if len(types) == 2 or powers[2] == "Egg":
    #     """ RULE 7-E
    #     Type 2 > Type i for all other 15 <= i < 33
    #     Formally:
    #         TypeI - Type2 <= -1, if ind(TypeI) < ind(Type2)
    #         TypeI - Type2 <=  0, if ind(TypeI) >= ind(Type2)
    #     """
    #     # print(powers)
    #     eleI = list(range(15, 33))
    #     eleI.remove(ele1)
    #     eleI.remove(ele2)
    #     eleI = np.array(eleI)
    #     A[eleI] -= data.MATRIX[ele2]
    #     b[eleI[eleI < ele2]] = -1
    
    # else:
    """ RULE 7
    Type 2 >  Type 3  if Type 3 comes before Type 2
    Type 2 >= Type 3  otherwise
    Formally:
        Type3 - Type2 <= -1, if ind(Type3) < ind(Type2)
        Type3 - Type2 <=  0, if ind(Type3) >= ind(Type2)
    """
    A[ele3] -= data.MATRIX[ele2]
    if ele3 < ele2:
        b[ele3] = -1

    """ Rule 8
    Type 3 > Type i for all 15 <= i < 33
    Formally:
        TypeI - Type3 <= -1, if ind(TypeI) < ind(Type3)
        TypeI - Type3 <=  0, if ind(TypeI) >= ind(Type3)
    """
    eleI = list(range(15, 33))
    eleI.remove(ele1)
    eleI.remove(ele2)
    eleI.remove(ele3)
    eleI = np.array(eleI)
    A[eleI] -= data.MATRIX[ele3]
    b[eleI[eleI < ele3]] = -1

    """ Rule 9
    Count of Fillings <= 5 and Count of Condiments <= 4
    Formally:
        Filling    <=  5
        Condiment  <=  4
        -Filling   <= -1
        -Condiment <= -1
    """
    b[-4] = 5
    b[-3] = 4
    b[-2] = -1
    b[-1] = -1

    """ Rule 10
    Apply the bonus from flavor to the corresponding power.
    e.g. if bonus is Power 2: Power1 > Power 2 + 100, Power 2 + 100 > Power 3
    Formally:
        if Bonus == Power 1:
            Power2 - Power1 <= ( 100 | 99  )
        if Bonus == Power 2:
            Power2 - Power1 <= (-100 | -101)
            Power3 - Power2 <= ( 100 | 99  )
        if Bonus == Power 3:
            Power3 - Power2 <= (-100 | -101)
            PowerI - Power3 <= ( 100 | 99  )
        else:
            Bonus  - Power3 <= (-100 | -101)
    """
    bonus = data.FLAVOR_BONUS[flavors[0]][flavors[1]]
    if bonus == powers[0]:
        b[power2] += 100
    elif bonus == powers[1]:
        b[power2] -= 100
        b[power3] += 100
    elif bonus == powers[2]:
        b[power3] -= 100
        b[powerI] += 100
    else:
        powerB = data.row2ind[bonus]
        b[powerB] -= 100

    return A, b

def reorderTypes(types):
    return [types[0], types[2], types[1]]

def getFlavors(bonus):
    for flav1, second in data.FLAVOR_BONUS.items():
        for flav2, power in second.items():
            if power == bonus:
                yield flav1, flav2

def buildQuery(powers, types, lv2=0):
    # types = reorderTypes(types)
    A, b = [], []

    starts = []
    for p, t in zip(powers, types):
        if p == 'Any':
            p = data.POWERS[:]
        else:
            p = p.split('|')
        if t == 'Any':
            t = data.TYPES[:]
        else:
            t = t.split('|')
        starts.append((p, t))

    seeds = []
    if lv2 == 0 or lv2 == 3:
        for o in (starts, starts[::-1]):
            for g in (o, o[1:] + o[:1], o[2:] + o[:2]):
                seeds.append(zip(*g))
    elif lv2 == 1:
        seeds.append(zip(*starts))
        seeds.append(zip(starts[0], starts[2], starts[1]))
    elif lv2 == 2:
        seeds.append(zip(*starts))
        seeds.append(zip(starts[1], starts[0], starts[2]))
    
    for p, t in seeds:
        for p0 in p[0]:
            for t0 in t[0]:
                for p1 in p[1]:
                    for t1 in t[1]:
                        for p2 in p[2]:
                            if len({p0, p1, p2}) < 3:
                                continue
                            for t2 in t[2]:
                                if len({t0, t1, t2}) < 3:
                                    continue
                                for flavors in getFlavors(p0):
                                    A0, b0 = buildExactQuery(
                                        flavors, [p0, p1, p2], [t0, t2, t1],
                                        lv2
                                    )
                                    A.append(A0)
                                    b.append(b0)

    print("Seed of size", len(A))
    return A, b

def findRecipes(A, b):
    lowest = None
    for i in intSimplex(A, b, data.npieces):
        if lowest is None or np.sum(i) <= lowest:
            printRecipe(i)
            lowest = np.sum(i)

A, b = buildQuery(
    ['Encounter', 'Catching', 'Any'],
    ['Dark|Flying', 'Dark|Flying', 'Any'],
    0
)
findRecipes(A, b)
