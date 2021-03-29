""" For computing BSCs """

def pw_bsc(n, m, mode="tm"):
    if mode.lower() == "tm":
        return .5
    if mode.lower() == "te":
        if m == 1: return -1j / 2
        if m == -1: return 1j / 2
    return 0

def make_pw_bscs(n_max):
    bscs = {"TM": {}, "TE": {}}
    for mode in bscs:
        for n in range(1, n_max + 1):
            for m in [-1, 1]:
                bscs[mode][(n, m)] = pw_bsc(n, m, mode=mode.lower())
    return bscs