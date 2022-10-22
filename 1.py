#first question 
#A
#The presence of noise affects image quality. Image denoising process attempts to reconstruct a noiseless image and improve its quality. Denoising an image with additive white Gaussian noise (AWGN) is a challenging process. Parameters such as noise mean and variance provide noise characteristics of AWGN
import cv2
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
# original image
f = cv2.imread('/content/th.jpg', 0)

cv2_imshow(f)
print("original image")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("#------------------------------#")

# create gaussian noise
x, y = f.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y))

cv2_imshow(n)
print("THIS IS OUR NOISE")
cv2.waitKey(0)
cv2.destroyAllWindows()

# display the probability density function (pdf)
kde = gaussian_kde(n.reshape(int(x*y)))
dist_space = np.linspace(np.min(n), np.max(n), 100)
plt.plot(dist_space, kde(dist_space))
plt.xlabel('Noise pixel value'); plt.ylabel('Frequency')
plt.show()

# add a gaussian noise
g = f + n

cv2_imshow(g)
print("THIS IS OUR COMBINE IMAGE NOISE WITH NORMAL IMAGE ")
cv2.waitKey(0)
cv2.destroyAllWindows()

# display all
print("ALL 3 IMAGES ARE PRINTED AS FOLLOWS")
cv2_imshow(f)
cv2_imshow(n)
cv2_imshow(g)

cv2.waitKey(0)
cv2.destroyAllWindows()
#also you can use following method


#mean = np.mean(img_1)#we are finding mean
#std = np.std(img_1)#we are find standred devition
#ele = len(img_1)#total element

#def add_noise(data): # assume data shape is (batch,channel,time), but it can also support (batch,time), (batch,anything,whatever,channel,time)

    #time_axis = len(data.shape)-1
    #target_snr_db = 20
    
    #data_watts = data ** 2
    #sig_avg_watts = np.mean(data_watts, axis=time_axis)
    #sig_avg_db = 10 * np.log10(sig_avg_watts)
    #noise_avg_db = sig_avg_db - target_snr_db
    #noise_avg_watts = 10 ** (noise_avg_db / 10)
    #mean_noise = 0
    #noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), data.shape) # <-- problem here
    # add noise to the original signal
    #noise_data = data + noise_volts
    #return noise_data

#x = np.random.normal(floor(mean), floor(std), floor(ele))
#plt.plot(x[0,0])
#plt.show()

#y = add_noise(x)
#plt.plot(y[0,0])
#plt.show()



#B
from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
     original = cv2.imread("/content/th.jpg")
     compressed = cv2.imread("/content/th.jpg", 1)
     value = PSNR(original, compressed)
     print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()

#C


import cv2
import numpy as np
from scipy import signal

def cal_ssim(img1, img2):
    
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim,ssim_map

# Assuming single channel images are read. For RGB image, uncomment the following commented lines
img2_ = cv2.imread('/content/f5.jpg',0)
#img1 = cv2.cvtColor(img2_, cv2.COLOR_BGR2GRAY)
img2_2 = cv2.imread('/content/f5.jpg',0)
#img2 = cv2.cvtColor(img2_2, cv2.COLOR_BGR2GRAY)

ssim_index, ssim_map = cal_ssim(img1, img2)
print(ssim_index,ssim_map)


#D
#isotropic gaussian blurring
import cv2 as cv
from google.colab.patches import cv2_imshow
img = cv.imread("/content/f5.jpg")
blur = cv.GaussianBlur(img,(5,5),0)
cv2_imshow(blur)