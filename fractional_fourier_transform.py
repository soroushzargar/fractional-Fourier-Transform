import numpy as np
import scipy.linalg as LinAlg


class FractionalFourierTransform:
    fourierMatrix = None
    signalLen = None

    def __init__(self):
        super().__init__()

    def makeFourierMatrix(self):
        N = self.signalLen
        self.fourierMatrix = N * np.array([[
            np.exp(
                1j * ((-2.0 * np.pi) * (ii) * (jj)) / np.fix(N)
                ) / np.fix(N)
            for ii in range(N)]
            for jj in range(N)
            ], dtype=np.complex128
        )
        return

    def frft(self, y, alpha=1, n=None, axis=-1, norm=None):
        # # Since the transformation is Periodic with p=4
        # alpha = np.remainder(alpha, 4.0)

        # To perform complex arithmetic, y should be complex
        complexInput = y.copy().astype(np.complex128)

        # Creating output container
        output = np.zeros_like(complexInput)

        # Getting Length of the Signal
        self.signalLen = complexInput.shape[0]
        # Checking the Fourier Transform Matrix
        if self.fourierMatrix is None:
            self.makeFourierMatrix()
        elif self.fourierMatrix.shape[0] != self.signalLen:
            self.makeFourierMatrix()

        # Taking the fourier matrix to fractional power
        fractionalFTMatrix = LinAlg.fractional_matrix_power(
            self.fourierMatrix, alpha
        )
        output = np.matmul(
            fractionalFTMatrix,
            np.transpose(complexInput)
        )
        return output
    
    
    def rfrft(self, y, alpha=1, n=None, axis=-1, norm=None):
        # # Since the transformation is Periodic with p=4
        # alpha = np.remainder(alpha, 4.0)

        # To perform complex arithmetic, y should be complex
        complexInput = y.copy().astype(np.complex128)

        # Creating output container
        output = np.zeros_like(complexInput)

        # Getting Length of the Signal
        self.signalLen = complexInput.shape[0]
        # Checking the Fourier Transform Matrix
        if self.fourierMatrix is None:
            self.makeFourierMatrix()
        elif self.fourierMatrix.shape[0] != self.signalLen:
            self.makeFourierMatrix()

        # Taking the fourier matrix to fractional power
        fractionalFTMatrix = LinAlg.fractional_matrix_power(
            self.fourierMatrix, alpha
        )
        output = np.matmul(
            fractionalFTMatrix,
            np.transpose(complexInput)
        )
        SymmetricOutput = output[:self.signalLen//2 + 1]
        return SymmetricOutput
