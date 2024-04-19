class Foo:
    a : float
    b : int

def struct_output(x : In[float], y : In[int]) -> Foo:
    foo : Foo
    foo.a = x + y * x
    foo.b = y - x
    return foo

d_struct_output = rev_diff(struct_output)

# def d_struct_output(x : In[float], _dx : Out[float], y : In[int], _dy : Out[float], _dout: In[Foo]) -> void:
#     _t_float : Array[float, 1]
#     _stack_ptr_float : int
#     _t_int : Array[int, 1]
#     _stack_ptr_int : int
#     foo : Foo
#     _dfoo : Foo
#     (_dfoo).a = 0

#     (_t_float)[_stack_ptr_float] = (foo).a
#     _stack_ptr_float = (_stack_ptr_float) + ((int)(1))
#     foo.a = x + y * x

#     _t_int)[_stack_ptr_int] = (foo).b
#     _stack_ptr_int = (_stack_ptr_int) + ((int)(1))
#     (foo).b = float2int((int2float(y)) - (x))

#     _dfoo = (_dfoo) + (_dreturn)

#     # backpropagate, ignore cos foo.b is int 
#     # ...

#     _stack_ptr_float = (_stack_ptr_float) - ((int)(1))
#     (foo).a = (_t_float)[_stack_ptr_float]
#     (_dfoo).a = (float)(0.0) #???

#     _adj_0 : float
#     _adj_0 = _dfoo.a
#     _adj_1 : float
#     _adj_1 = _dfoo.a * int2float(y)
#     (_dfoo).a = (float)(0.0)
#     _dx = (_dx) + (_adj_0)
#     _dx = (_dx) + (_adj_1)
