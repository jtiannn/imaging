import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import cv2
from scipy import signal
import skimage as sk
import skimage.io as skio
from skimage import color
from scipy.misc import imresize
import scipy.stats as st
from skimage.filters import gaussian_filter
from skimage.exposure import rescale_intensity

#From stackoverflow
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def hybrid_image(im1, im2, s1, s2):
	low = gaussian_filter(im2, s1, mode='reflect')
	high = im1 - gaussian_filter(im1, s2, mode='reflect')
	hybrid = (low + high) / 2
	hybrid = rescale_intensity(hybrid, in_range=(0, 1), out_range=(0, 1))
	return hybrid, high, low
## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
def fourier(gray_image):
	return np.log(np.abs(np.fft.fftshift(np.fft.fft2(color.rgb2gray(gray_image)))))

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
def gaussian_stack(im, n=5):
	s = []
	gauss_im = im
	for i in range(n):
		gauss_im = gaussian_filter(gauss_im, 2**i, mode='reflect')
		s.append(gauss_im)
	return s

def laplacian_stack(im, gauss):
	s = []
	for g in gauss:
		lap_im = im - g
		s.append(lap_im)
	return s