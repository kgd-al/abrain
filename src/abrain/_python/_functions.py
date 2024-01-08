import numpy as np

functions = dict(
    ssgn=np.vectorize(lambda x: (
        np.exp(-(x+1)**2)-1
        if x < -1 else
        1 - np.exp(-(x-1)**2)
        if x > 1 else
        0
    )),

    id=lambda x: x,
    abs=lambda x: np.fabs(x),
    sin=lambda x: np.sin(2*x),
    step=lambda x: np.heaviside(x, 0),
    gaus=lambda x: np.exp(-6.25*x*x),
    ssgm=lambda x: 1 / (1 + np.exp(-4.9*x)),
    bsgm=lambda x: 2 / (1 + np.exp(-4.9*x)) - 1,
)
