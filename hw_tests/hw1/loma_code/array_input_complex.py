# complex test for input array...
def array_input_complex(x : In[Array[float]]) -> float:
    a : int = 5
    b : int = 4
    # return x[0] + x[a-b]
    z0 : float = x[0] + x[1]
    z1 : float = x[0] - x[1]
    z2 : float = x[0] * x[1]
    z3 : float = x[0] / x[1]
    z4 : float = z0 + z1 + z2 + z3
    z5 : float = sin(x[0])
    z6 : float = cos(x[0])
    z7 : float = sqrt(x[0])
    z8 : float = pow(x[0], x[1])
    z9 : float = exp(x[0])
    z10 : float = log(x[0] + x[1])
    z11 : float = z4 + z5 + z6 + z7 + z8 + z9 + z10
    return z11

d_array_input_complex = fwd_diff(array_input_complex)
