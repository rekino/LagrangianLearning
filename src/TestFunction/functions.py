import typing as t
import numpy.typing as npt
import numpy


class AbstractTestFunction(object):
    """
    Base class for test functions.

    Args:
        np: An object that mimics the Numpy interface. Replace this object if you want to
            make the test function differentiable against some automatic differentiation
            engine.
    """
    np = numpy
    
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Computes the test function for x.

        Args:
            x (numpy.typing.ArrayLike): An ArrayLike object that the test function is computed on.

        Returns:
            numpy.typing.ArrayLike: The value of the test function on x.
        """
        pass

class EuclideanBumpFunction(AbstractTestFunction):
    """
    A radial test function which is supported on the interval [-1, 1].

    Args:
        np: An object that mimics the Numpy interface. Replace this object if you want to
            make the test function differentiable against some automatic differentiation
            engine.
    """
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self.np.where(self.np.abs(x) > 1, 0, self.np.exp(-1 / (1 - x**2)))

class WrapperTestFunction(AbstractTestFunction):
    """
    A test function that wraps another function.

    Args:
        np: This object is ignored. It is assumed that the function is a smooth test function
        and is differentiable against some automatic differentiation engine.
    """
    fun: t.Callable[[npt.ArrayLike], npt.ArrayLike]

    def __init__(self, fun: t.Callable[[npt.ArrayLike], npt.ArrayLike]) -> None:
        super().__init__()
        self.fun = fun
    
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self.fun(x)
