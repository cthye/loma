def int_input(x : In[float], y : In[int]) -> float:
    z : int = 5
    return z * x + y - 1

d_int_input = rev_diff(int_input)

# def d_int_input(x: In[float], _dx: Out[float], y: In[int], _dy: Out[int], _dout: [float]) -> void:
#     z: int = 5
#     _dx += z * _dout
#     _dy = 0    