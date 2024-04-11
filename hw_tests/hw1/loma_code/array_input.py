def array_input(x : In[Array[float]]) -> float:
    a : int = 5
    b : int = 4
    # return x[0] + x[a-b]
    return x[0] + x[1]
    # return x[0] - x[1]
    # return x[0] * x[1]
    # return x[0] / x[1]
    # return sin(x[0])
    # return sin(x[0]+x[1])



d_array_input = fwd_diff(array_input)


# def d_array_input(x : In[Array[_dfloat]]) -> _dfloat:
        # return make__dfloat((((x)[(int)(0)]).val) + (((x)[(int)(1)]).val),(((x)[(int)(0)]).dval) + (((x)[(int)(1)]).dval))