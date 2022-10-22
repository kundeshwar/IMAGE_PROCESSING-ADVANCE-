#comparing psnr with soft thresholding and hard (this is for gray images)
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io
img = skimage.io.imread("/content/f5.jpg")#reading image
img = skimage.img_as_float(img)#convert image as float

sigma = 0.1 #noise std
imgn = random_noise(img, var=sigma**2) #adding noise

sigma_est = estimate_sigma(imgn, average_sigmas=True)  #noise Estimation

#Denoising using Bayes for hard
img_b = denoise_wavelet(imgn,method="BayesShrink",mode="hard", wavelet_levels=3, wavelet='bior6.8', rescale_sigma=True)

#Denoising using Visushrinks for soft
img_v = denoise_wavelet(imgn, method="VisuShrink", mode="soft", sigma=sigma_est/3, wavelet_levels=5,wavelet='bior6.8', rescale_sigma=True)

#finding PENR
psnr_noise = peak_signal_noise_ratio(img, imgn)
psnr_bayes = peak_signal_noise_ratio(img, img_b)
psnr_visu = peak_signal_noise_ratio(img, img_v)

#ploting images
plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
plt.imshow(img, cmap=plt.cm.gray)
plt.title("original image", fontsize=30)

plt.subplot(2,2,2)
plt.imshow(imgn, cmap=plt.cm.gray)
plt.title("noise image", fontsize=30)

#printing psnr
print("PENR [ORIGINAL Vs. NOISY]", psnr_noise)
print("PENR [ORIGINAL Vs. DENISED(soft)]", psnr_visu)
print("PENR [ORIGINAL Vs. DENOISED(HARD)]", psnr_bayes)

