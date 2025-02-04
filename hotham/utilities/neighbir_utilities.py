import numpy as np


def find_remain_index(i, j, S):
    remain_index = []
    neighborlist = []
    for index in range(len(i)):
        ijs = f"{i[index]}-{j[index]}-{S[index,0]}-{S[index,1]}-{S[index,2]}"
        ijs_inv = f"{j[index]}-{i[index]}-{-S[index,0]}-{-S[index,1]}-{-S[index,2]}"
        if (ijs not in neighborlist) and (ijs_inv not in neighborlist):
            neighborlist.append(ijs)
            remain_index.append(index)
    return remain_index


def find_remain_index_2(i, j, S, AtomNumber):
    remain_index = []
    neighborlist = []
    for index in range(len(i)):
        ijs = f"{i[index]}-{j[index]}-{S[index,0]}-{S[index,1]}-{S[index,2]}"
        ijs_inv = f"{j[index]}-{i[index]}-{-S[index,0]}-{-S[index,1]}-{-S[index,2]}"
        if (ijs not in neighborlist) and (ijs_inv not in neighborlist):
            atomnumber_i = AtomNumber[i[index]]
            atomnumber_j = AtomNumber[j[index]]
            if atomnumber_i <= atomnumber_j:
                neighborlist.append(ijs)
                remain_index.append(index)
    return remain_index


def find_inverse_index(I, J, S):
    index_inv = {}
    for index in range(len(I)):
        i, j = I[index], J[index]
        s1, s2, s3 = S[index]
        ijs = (i, j, s1, s2, s3)
        ijs_inv = (j, i, -s1, -s2, -s3)

        # index_inv[ijs] = index_inv.setdefault(ijs, [])+[index]
        index_inv[ijs] = [index]+index_inv.setdefault(ijs, [])
        index_inv[ijs_inv] = index_inv.setdefault(ijs_inv, [])+[index]

        # key = tuple(sorted((ijs, ijs_inv)))
        # index_inv[key] = index_inv.setdefault(key, [])+[index]
        # if ijs == ijs_inv:
        #     index_inv[key] = index_inv.setdefault(key, [])+[index]
    # return torch.tensor(list(index_inv.values())).T
    return np.array(sorted(index_inv.values()))[:, 1]
