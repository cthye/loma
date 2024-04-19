def array_input(x : In[Array[float]]) -> float:
    return 2 * x[0] + 3 * x[1] * x[1]

d_array_input = rev_diff(array_input)

# def d_array_input(x : In[float], _dx : Out[float], _dout : Out[float]]):
#     _dx[0] += 2 * _dout
#     _dx[1] += 2 * 3 * _dout * x[1]