import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	ary_a, ary_b = np.array(a), np.array(b)
	try:
		return np.dot(ary_a, ary_b)
	except ValueError:
	    return -1