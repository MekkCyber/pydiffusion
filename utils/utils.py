def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)