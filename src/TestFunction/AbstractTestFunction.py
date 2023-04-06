import numpy
import numpy.typing as npt
import typing as t


class AbstractTestFunction(object):

    np = numpy

    def __init__(self, pesudo_numpy=numpy) -> None:
        """
        Constructor for the test function.

        Args:
            pesudo_numpy: An object that mimics the Numpy interface. Replace this object if
             you want to make the test function differentiable against some automatic differentiation
             engine.
        """
        self.np = pesudo_numpy
    
    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Computes the test function for x.

        Args:
            x (numpy.typing.ArrayLike): An ArrayLike object that the test function is computed on.

        Returns:
            numpy.typing.ArrayLike: The value of the test function on x.
        """
        pass
