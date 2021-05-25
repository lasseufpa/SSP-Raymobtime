
import numpy as np
from numpy import linalg


def hyst_quantize_real(x, std=1):

    # normalized standard deviation of the thresholds

    N, T = x.shape

    norm = linalg.norm(x, 'fro')

    t1, t2 = norm / np.sqrt(N*T)*np.sort(np.random.normal(0, std, 2))

    output = np.zeros((N, T))
    output[:, 0] = np.sign(x[:, 0])
    # first data vector is not affected for simplicity

    thresholds = np.zeros(N)
    for idx in range(1, T):
        # fill with t1
        thresholds[:] = t1

        # change only where is needed
        isPreviousOutputsNegative = output[:, idx-1] < 0
        thresholds[isPreviousOutputsNegative] = t2

        output[:, idx] = np.sign(x[:, idx] - thresholds)

    return output


def hysteresis_quantize(x, std=1):
    re, im = [hyst_quantize_real(val, std) for val in [np.real(x), np.imag(x)]]
    return re + 1j * im


if __name__ == '__main__':
    a = np.random.randn(9, 9)
    b = np.random.randn(9, 9)

    c = (a + 1j*b) / np.sqrt(2)

    print(hysteresis_quantize(c))

# scatter(input(:), output(: ))
