import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    ary_a, ary_b = np.asarray(a), np.asarray(b)

    if ary_a.ndim != 2 or ary_b.ndim != 2 or ary_a.shape[1] != ary_b.shape[0]:
        return -1

    return (ary_a @ ary_b).tolist()