import unittest
import numpy as np
from sinusoidal_regression.estimator import SinusoidalRegression

class SingleWaveRegression(unittest.TestCase):
    def test_regression(self):
        N = 5000  # number of data points
        T = np.linspace(0, 4 * np.pi, N)
        TRUE_AMPLITUDE = 3.0
        TRUE_FREQUENCY = 1.2
        TRUE_PHASE = 0.1
        TRUE_MEAN = 0.5
        y = TRUE_AMPLITUDE * np.sin(TRUE_FREQUENCY*T + TRUE_PHASE) + TRUE_MEAN + np.random.randn(N)
        m = SinusoidalRegression()
        m.fit_sinusoidal_model(T, y)
        std_mle, frequency_mle, phase_mle, mean_mle = m.params_
        self.assertAlmostEqual(std_mle, TRUE_AMPLITUDE, places=1)
        self.assertAlmostEqual(phase_mle, TRUE_PHASE, places=1)
        self.assertAlmostEqual(mean_mle, TRUE_MEAN, places=1)
        self.assertAlmostEqual(frequency_mle, TRUE_FREQUENCY, places=1)