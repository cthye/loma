def call_log(x : In[float]) -> float:
    return log(x)

d_call_log = fwd_diff(call_log)

# def call_log(x : In[float], y: In[float]) -> float:
#     return log(x + y)


# d_call_log = fwd_diff(call_log)
