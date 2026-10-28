import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # Your code here
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (2, 2):
        raise ValueError(f"expected a 2x2 matrix, got shape {mat.shape}") 
    
    # below is No. matrix_rank is not precise.
    # also np.linalg.det is a No.
    # if np.linalg.matrix_rank(mat) != 2:
    #    return None

    try:
        return np.linalg.inv(mat).tolist()
    except np.linalg.LinAlgError:
        return None
    