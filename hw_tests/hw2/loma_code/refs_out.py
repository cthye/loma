def refs_out(x : In[float], y : Out[float]):
    y = x * x

d_refs_out = rev_diff(refs_out)

# def d_refs_Out(x: In[float], _dx: Out[float], y: In[float]):
#     _dx += 2 * x * _dy
