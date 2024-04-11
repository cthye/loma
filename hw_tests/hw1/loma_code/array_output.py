def array_output(x : In[float], y : Out[Array[float]]):
    y[0] = x * x
    y[1] = x * x * x

d_array_output = fwd_diff(array_output)

# def d_array_output(x : In[_dfloat], y : Out[Array[_dfloat]]) -> void:
#         (y)[(int)(0)] = make__dfloat(((x).val) * ((x).val),(((x).val) * ((x).dval)) + (((x).val) * ((x).dval)))
#         (y)[(int)(1)] = make__dfloat((((x).val) * ((x).val)) * ((x).val),((((x).val) * ((x).val)) * ((x).dval)) + (((x).val) * ((((x).val) * ((x).dval)) + (((x).val) * ((x).dval)))))

