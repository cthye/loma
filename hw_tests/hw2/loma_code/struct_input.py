class Foo:
    x : float
    y : int
    z : float

def struct_input(foo : In[Foo]) -> float:
    w : int = 5
    return w * foo.x + foo.y + foo.x * foo.z - 1

d_struct_input = rev_diff(struct_input)


# def d_struct_input(foo : In[Foo], _dfoo: Out[Foo], _dout: In[float]) -> void:
#     _dfoo.x  += w * _dout
#     _dfoo.x  += foo.z * _dout
#     _dfoo.z  += foo.x * _dout