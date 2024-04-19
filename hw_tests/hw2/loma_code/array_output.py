def array_output(x : In[float], y : Out[Array[float]]):
    y[0] = x * x
    y[1] = x * x * x

d_array_output = rev_diff(array_output)

# def d_array_output(x : In[float], _dx : Out[float], _dy : In[Array[float]]):
#     _dx += 2 * x * _dy[0]
#     _dx += 3 * x * x * _dy[1]