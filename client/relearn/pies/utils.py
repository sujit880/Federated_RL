import numpy as np


def compare_weights(weights1, weights2):
    w1 = weights1
    w2 = weights2
    names1 = []
    names2 = []
    if(len(w1) != len(w2)):
        #print("Models arcitecture not same: ",len(w1), len(w2) )
        return False
    for x in (w1):
        names1.append(x)
    for y in (w2):
        names2.append(y)
    if(len(w1) == len(names1)):
        count = 0
        for j in range(len(names1)):
            if(np.array_equal(w1[names1[j]], w2[names2[j]])):
                count += 1
                continue
            else:
                #print("Weights are different")
                return False
        print("Total dict:", count)
        return True
