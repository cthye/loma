def divide(x : In[float], y : In[float]) -> float:
    return x / y

# def divide(x : In[Array[float]]) -> float:
#     return x[0] / x[1]

d_divide = fwd_diff(divide)
