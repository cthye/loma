def int_assign(x : In[float], y : In[int]) -> float:
    z : int = y
    w : float = z * x
    z = 6
    return w * z * x + y - 1

d_int_assign = rev_diff(int_assign)

# def d_int_assign(x : In[float], _dx: Out[float], y : In[int], _dy: Out[Int], _dout: In[float]) -> void:
#     _t_int : Array[int, 1]
#     _stack_ptr_int : int
#     z : int = y
#     w : float = z * x
#     _dw : float
#     (_t_int)[_stack_ptr_int] = z
#     _stack_ptr_int = (_stack_ptr_int) + 1
#     z = 6
#     #backpropogate return
#     _dw += z * x
#     _dx += w * z 
#     # _dx = y * x * 6

#     #backpropogate assign z = 6
#     _stack_ptr_int = (_stack_ptr_int) - 1
#     z = (_t_int)[_stack_ptr_int]
#     # nothing happen for z is int
#     #backpropogate declare w = z * x
#     _dx += z * _dw
#     # _dx = y * x * 6 + y * 6 * x

#     #backpropogate declare z = y
#     # nothing happen

    

