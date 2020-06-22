import itertools

def GetPermutations(paramsDic):
    s = list()
    keyToIndexDic = {}
    index = 0
    for key in paramsDic:
        keyToIndexDic[key] = index
        index += 1
        s.append(paramsDic[key])

    permutations = list(itertools.product(*s))
    return permutations, keyToIndexDic


# ------------------ Bayesian optimization --------------------------
