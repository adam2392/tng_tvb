# coding=utf-8

import numpy as np
# Default model parameters

# MODEL params in milliseconds
WINSIZE_FRAG = 250
STEPSIZE_FRAG = 125
WINSIZE_SPEC = 5000
STEPSIZE_SPEC = 2500
MTBANDWIDTH = 4 		# multitaper FFT bandwidth
WAVELETWIDTH = 6		# WIDTH of our wavelets
RADIUS = 1.5 			# perturbation radius

# how can we inject noise into our models?
WHITE_NOISE = "White"
COLORED_NOISE = "Colored"
NOISE_SEED = 42

TIME_DELAYS_FLAG = 0.0
MAX_DISEASE_VALUE = 1.0 - 10 ** -3
