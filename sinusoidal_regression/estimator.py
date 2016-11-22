from scipy.optimize import leastsq
import numpy as np

def fit_sinusoidal_model(t, y):
    mean_guess = np.mean(y)
    std_guess = 3*np.std(y)/(2**0.5)
    phase_guess = 0
    f = lambda p: p[0]*np.sin(t + p[1]) + p[2] - y
    std_mle, phase_mle, mean_mle = leastsq(f, [std_guess, phase_guess, mean_guess])[0]
    return std_mle, phase_mle, mean_mle

if __name__ == '__main__':
    N = 1000  # number of data points
    t = np.linspace(0, 4 * np.pi, N)
    y = 3.0 * np.sin(t + 0.001) + 0.5 + np.random.randn(N)  # create artificial data with noise
    print(fit_sinusoidal_model(t, y))