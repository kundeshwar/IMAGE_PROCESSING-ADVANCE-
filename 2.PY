
#below code is showing both hard thresholding and soft thresholding and also shows two method like bayes and visushrinks
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io
img = skimage.io.imread("/content/f5.jpg")#reading image
img = skimage.img_as_float(img)#convert image as float

sigma = 0.1#noise std
imgn = random_noise(img, var=sigma**2)#adding noise

sigma_est = estimate_sigma(imgn, average_sigmas=True)#noise Estimation

#Denoising using Bayes for hard
img_b = denoise_wavelet(imgn,method="BayesShrink",mode="hard",wavelet_levels=3, wavelet='bior6.8',rescale_sigma=True)

#Denoising using Visushrinks for soft
img_v = denoise_wavelet(imgn, method="VisuShrink", mode="soft", sigma=sigma_est/3, wavelet_levels=5,wavelet='bior6.8',rescale_sigma=True)

