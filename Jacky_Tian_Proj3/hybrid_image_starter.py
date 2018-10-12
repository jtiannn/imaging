import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import cv2
from scipy import signal

# First load images

# high sf
im1 = plt.imread('DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

kernel = np.array([[1.0, 4, 7, 4, 1],
				  [4, 16, 26, 16, 4],
				  [7, 26, 41, 26, 7],
				  [4, 16, 26, 16, 4],
				  [1, 4, 7, 4, 1]])
for i in range(len(kernel)):
	for j in range(len(kernel[0])):
		kernel[i][j] = float(kernel[i][j])/273.0

def normalize(im):
	max_num = 0
	for a in im:
		for b in a:
			for c in b:
				max_num = max(max_num, c)
	for a in im:
		for b in a:
			for c in b:
				c = c/max_num
	return im

def hybrid_image(im1, im2, s1, s2):
	r = im1[:,:,0]
	g = im1[:,:,1]
	b = im1[:,:,2]
	ab = signal.convolve2d(b, kernel, mode='same')
	ag = signal.convolve2d(g, kernel, mode='same')
	ar = signal.convolve2d(r, kernel, mode='same')
	low = np.dstack([ar, ag, ab])
	r2 = im2[:,:,0]
	g2 = im2[:,:,1]
	b2 = im2[:,:,2]
	ab2 = signal.convolve2d(b2, kernel, mode='same')
	ag2 = signal.convolve2d(g2, kernel, mode='same')
	ar2 = signal.convolve2d(r2, kernel, mode='same')
	high = np.clip(np.subtract(im2, np.dstack([ar2, ag2, ab2])), 0, 1)
	return (low + high) / 2
## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 0
sigma2 = 0
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.imshow(hybrid)
plt.show

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
pyramids(hybrid, N)