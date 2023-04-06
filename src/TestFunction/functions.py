import typing as t
import numpy.typing as npt
from .AbstractTestFunction import AbstractTestFunction


class EuclideanBumpFunction(AbstractTestFunction):

    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self.np.where(self.np.abs(x) > 1, 0, self.np.exp(-1 / (1 - x**2)))

class LambdaTestFunction(AbstractTestFunction):

    fun: t.Callable[[npt.ArrayLike], npt.ArrayLike]

    def __init__(self, fun: t.Callable[[npt.ArrayLike], npt.ArrayLike]) -> None:
        super().__init__()
        self.fun = fun
    
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self.fun(x)