import unittest

import numpy as np

import src.TestFunction as tf

class TestTestFunctions(unittest.TestCase):

    def test_Psi(self):
        r = np.linspace(-1, 1, 3)
        out = tf.Psi(r)
        self.assertTrue(np.all(np.isclose(out, np.array([0, np.exp(-1), 0]))))
    
    def test_lambda(self):
        r = np.linspace(-np.pi, np.pi, 3)
        test_function = tf.WrapperTestFunction(lambda _x: np.where(np.abs(_x) >= np.pi/2, 0, np.exp(-np.tan(_x)**2)))
        out = test_function(r)
        self.assertTrue(np.all(np.isclose(out, np.array([0, 1, 0]))))

if __name__ == '__main__':
    unittest.main()