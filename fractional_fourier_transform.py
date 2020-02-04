import numpy as np
import scipy.linalg as LinAlg


class FractionalFourierTransform:
    precalculated_fourierMatrix = None
    precalculated_fractionalTransform = None
    precalculated_inverseTransform = None
    signalLen = None
    alpha = None

    def __init__(self, alpha=1, signalLen=None):
        #super().__init__()

        if signalLen is not None:
            self.signalLen = signalLen

        if alpha is not None:
            self.alpha = alpha

    def makeFourierMatrix(self):
        print("creating Fourier Matrix on dimension: ", self.signalLen)
        N = self.signalLen
        self.precalculated_fourierMatrix = N * np.array([[
            np.exp(
                1j * ((-2.0 * np.pi) * (ii) * (jj)) / np.fix(N)
                ) / np.fix(N)
            for ii in range(N)]
            for jj in range(N)
            ], dtype=np.complex128
        )
        return

    def fft(self, y, n=None, axis=-1, norm=None):
        # This line ensures that the input is a numpy array
        y = np.array(y)

        if n is None:
            n = y.shape[axis]

        y = y[:n]

        if self.signalLen is None:
            self.signalLen = n

        # Class Initial Checks
        if self.signalLen != n:
            self.signalLen = n
            self.precalculated_fourierMatrix = None
            self.precalculated_fractionalTransform = None

        if self.precalculated_fourierMatrix is None:
            self.makeFourierMatrix()
        if self.precalculated_fractionalTransform is None:
            print("Creating Fractional Matrix")
            self.precalculated_fractionalTransform = \
                LinAlg.fractional_matrix_power(
                    self.precalculated_fourierMatrix, self.alpha
                )

        # On this stage, the precalculated matrices are ready

        # To perform complex arithmetic, y should be complex
        complexInput = y.copy().astype(np.complex128)

        # Creating output container
        output = np.zeros_like(complexInput)

        output = np.matmul(
            self.precalculated_fractionalTransform,
            complexInput
        )
        return output

    def rfft(self, y, n=None, axis=-1, norm=None):
        result = self.fft(y, n, axis, norm)
        N = result.shape[axis]
        return result[:(N//2 + 1)]

    def ifft(self, y, n=None, axis=-1, norm=None):
        # This line ensures that the input is a numpy array
        y = np.array(y)

        if n is None:
            n = y.shape[axis]

        y = y[:n]

        if self.signalLen is None:
            self.signalLen = n

        # Class Initial Checks
        if self.signalLen != n:
            self.signalLen = n
            self.precalculated_fourierMatrix = None
            self.precalculated_inverseTransform = None

        if self.precalculated_fourierMatrix is None:
            self.makeFourierMatrix()
        if self.precalculated_inverseTransform is None:
            self.precalculated_inverseTransform = \
                LinAlg.fractional_matrix_power(
                    self.precalculated_fourierMatrix, -1 * self.alpha
                )

        # On this stage, the precalculated matrices are ready

        # To perform complex arithmetic, y should be complex
        complexInput = y.copy().astype(np.complex128)

        # Creating output container
        output = np.zeros_like(complexInput)

        output = np.matmul(
            self.precalculated_inverseTransform,
            complexInput
        )
        return output

    def irfft(self, y, n=None, axis=-1, norm=None):
        y_rev = None
        if self.signalLen % 2 == 1:
            y_rev = np.flip(y[1:], axis)
        else:
            y_rev = np.flip(y[1:-1], axis)
        yConventional = np.concatenate((y, y_rev), axis=axis)
        return self.ifft(yConventional, n, axis, norm).real
