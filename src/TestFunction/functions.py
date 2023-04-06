import numpy as np
import numpy.typing as npt


def euclidean_bump_function(x: npt.ArrayLike, abs=np.abs, exp=np.exp, where=np.where) -> npt.ArrayLike:
    return where(abs(x) > 1, 0, exp(-1 / (1 - x**2)))
