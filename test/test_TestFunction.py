import unittest

import numpy as np

import src.TestFunction as tf

class TestTestFunctions(unittest.TestCase):

    def test_Psi(self):
        r = np.linspace(-1, 1, 3)
        out = tf.Psi(r)
        self.assertTrue(np.all(np.isclose(out, np.array([0, np.exp(-1), 0]))))

if __name__ == '__main__':
    unittest.main()