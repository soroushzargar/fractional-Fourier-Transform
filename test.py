import librosa
import numpy as np
import fractional_fourier_transform as frft
import random

N = 1000
signal = np.array([random.uniform(0, 1) for i in range(N)])
fractionalTransformer = frft.FractionalFourierTransform()

# First Test: Identity Matrix:
print("""=========================================================
Test One: In this test the signal is given to the fractional
transform with ratio = 0, it is supposed to return the identity.
""")

testRes = fractionalTransformer.frft(
	y=signal, alpha=0
)

if np.allclose(testRes, signal):
	print("Identity Test Passed...!")
else:
	print("X: Identity Test Failed...!")
print("""=========================================================

""")

# Second Test: Conventional Fourier Matrix:
print("""=========================================================
Test Two: In this test the signal is given to the fractional
transform with ratio = 1, it is supposed to return the conventional
Fourier Transform of the input.
""")

testRes = fractionalTransformer.frft(
	y=signal, alpha=1
)

baseline = np.fft.fft(signal)

if np.allclose(testRes, baseline):
	print("Conventional Fourier Test Passed...!")
else:
	print("X: Conventional Fourier Test Failed...!")
print("""=========================================================

""")

# Third Test: Conventional Inverse Fourier Matrix:
print("""=========================================================
Test Three: In this test the signal is given to the fractional
transform with ratio = -1, it is supposed to return the conventional
inverse Fourier Transform of the input.
""")

testRes = fractionalTransformer.frft(
	y=signal, alpha=-1
)

baseline = np.fft.ifft(signal)

if np.allclose(testRes, baseline):
	print("Conventional inverse Fourier Test Passed...!")

else:
	print("X: Conventional inverse Fourier Test Failed...!")
print("""=========================================================

""")

# Fourth Test: Revertibity on different fractions of Fourier Matrix:
print("""=========================================================
Test Four: In this test the signal is given to different ratios of
the fourier transform and the result will be returned to the first
time domain. It is supposed to be revertible.
""")

failList = []
for intAlpha in range(20):
	alpha = intAlpha/10.0
	ftRes = fractionalTransformer.frft(
		signal, alpha
	)
	iftRes = fractionalTransformer.frft(
		ftRes, -1*alpha
	)
	if not np.allclose(signal, iftRes):
		failList.append(alpha)
	

if len(failList) == 0:
	print("Fractional revertibility Test Passed...!")

else:
	print("X: Fractional revertibility Test Failed...!")
	print("failure list is: ", failList)
print("""=========================================================

""")

