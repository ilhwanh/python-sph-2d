import math

def kernel_cubic_b_spline(d, h):
    
    if d == 1:
        C = 2/3
    elif d == 2:
        C = 10 / (7 * math.pi)
    elif d == 3:
        C = 1 / math.pi
    else:
        raise ValueError(f"Unsupported dimension {d}")
    factor = C / (h ** d)

    def original(r):
        if r < 1:
            return factor * (1 - 3/2 * (r ** 2) + 3/4 * (r ** 3))
        elif r <= 2:
            return factor * (1/4 * ((2 - r) ** 3))
        else:
            0

    def first_derivate(r):
        if r < 1:
            return factor * (3 * r + 9/4 * (r ** 2))
        elif r <= 2:
            return -factor * (3/4 * ((2 - r) ** 2))
        else:
            0

    def second_derivate(r):
        if r < 1:
            return factor * (3 + 9/2 * r)
        elif r <= 2:
            return factor * (3/2 * (2 - r))
        else:
            0
    
    return original, first_derivate, second_derivate