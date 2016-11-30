from scipy.optimize import leastsq, least_squares
import numpy as np
import pylab as plt
import seaborn as sns

class SinusoidalRegression(object):
    def fit_sinusoidal_model(self, t, y):
        mean_guess = np.mean(y)
        std_guess = 3*np.std(y)/(2**0.5)
        frequency_guess = 1
        phase_guess = 0
        f = lambda p: p[0]*np.sin(p[1]*t + p[2]) + p[3] - y
        std_mle, frequency_mle, phase_mle, mean_mle = leastsq(f, [std_guess, frequency_guess, phase_guess, mean_guess])[0]
        self.params_ = std_mle, frequency_mle, phase_mle, mean_mle

    def predict(self, t):
        std_mle, frequency_mle, phase_mle, mean_mle = self.params_
        y_predicted = std_mle * np.sin(frequency_mle*t + phase_mle) + mean_mle
        return y_predicted

if __name__ == '__main__':
    N = 5000  # number of data points
    T = np.linspace(0, 4 * np.pi, N)
    y = 3.0 * np.sin(T + 0.1) + 0.5 + np.random.randn(N)  # create artificial data with noise
    m = SinusoidalRegression()
    m.fit_sinusoidal_model(T, y)
    print(m.params_)
    y_predicted = m.predict(T)
    plt.plot(y, '.')
    plt.plot(y_predicted, label='after fitting', color='r')
    plt.legend()
    plt.show()

