def call_pow(x : float, y : float) -> float:
    return pow(x, y)

d_call_pow = fwd_diff(call_pow)
