import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import error
import math
import gpuctypes.opencl as cl
import cl_utils
import numpy as np

def test_atomic_add():
    with open('loma_code/atomic_add.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/atomic_add.so')

    c_func = getattr(lib, 'my_atomic_add')
    c_func.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    c_func.restype = None
    
    np.random.seed(seed=1234)
    n = 10000
    x = np.random.random(n).astype('f') / n
    z = ctypes.c_float(0)
    ctypes_x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.my_atomic_add(ctypes_x, ctypes.byref(z), n)
    assert abs(z.value - np.sum(x)) < 1e-3


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    test_atomic_add()