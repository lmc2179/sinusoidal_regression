import unittest
import numpy as np
from sinusoidal_regression.estimator import SinusoidalRegression

class SingleWaveRegression(unittest.TestCase):
    def test_regression(self):
        N = 5000  # number of data points
        T = np.linspace(0, 4 * np.pi, N)
        y = 3.0 * np.sin(T + 0.1) + 0.5 + np.random.randn(N)  # create artificial data with noise
        m = SinusoidalRegression()
        m.fit_sinusoidal_model(T, y)
        std_mle, phase_mle, mean_mle = m.params_
        self.assertAlmostEqual(std_mle, 3.0, places=1)
        self.assertAlmostEqual(phase_mle, 0.1, places=1)
        self.assertAlmostEqual(mean_mle, 0.5, places=1)