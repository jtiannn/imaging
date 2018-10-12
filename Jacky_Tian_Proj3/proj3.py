# CS194-26 (CS294-26): Project 3

# Jacky Tian

import numpy as np
import cv2
from scipy import signal
from scipy import sparse
import skimage as sk
import skimage.io as skio
from scipy.misc import imresize
from hybrid_image import *

def sharpen(fname, kernel):
	im = skio.imread(fname)
	im = sk.img_as_float(im)
	height = np.floor(im.shape[0] / 3.0).astype(np.int)
	# separate color channels
	r = im[:,:,0]
	g = im[:,:,1]
	b = im[:,:,2]

	ab = signal.convolve2d(b, kernel, mode='same')
	ag = signal.convolve2d(g, kernel, mode='same')
	ar = signal.convolve2d(r, kernel, mode='same')

	smooth = np.dstack([ar, ag, ab])
	detail = np.clip(np.subtract(im, smooth), 0, 1)
	im_out = np.clip(np.add(im, detail), 0, 1)
	return im_out

def mask(im, m, side):
	ma = m
	if side != 'left':
		ma = 1 - m
	im_mask = [im[:,:,0]*ma, im[:,:,1]*ma, im[:,:,2]*ma]
	return np.dstack(im_mask)

def blend(im1, im2, msk):
	m = msk
	im1_stack = laplacian_stack(im1, gaussian_stack(im1))
	im2_stack = laplacian_stack(im2, gaussian_stack(im2))
	m_stack = gaussian_stack(m)
	im1_mask = []
	im2_mask = []
	for i in range(len(m_stack)):
		im1_mask.append(mask(im1_stack[i], m_stack[i], 'left'))
		im2_mask.append(mask(im2_stack[i], m_stack[i], 'right'))
	for i in range(len(im1_mask)):
		im1_mask[i] = im1_mask[i] + im2_mask[i]
	kernel_mask = gaussian_filter(m, 32, mode='reflect')
	im1_kernel = mask(gaussian_filter(im1, 32), kernel_mask, 'left')
	im2_kernel = mask(gaussian_filter(im2, 32), kernel_mask, 'right')
	im_blend = sum(im1_mask, sum([im1_kernel, im2_kernel]))
	im_blend = rescale_intensity(im_blend, in_range = (0, 1), out_range=(0, 1))
	return im_blend

def update_toy(im):
	rows = len(im)
	cols = len(im[0])
	im2var = np.zeros((rows, cols))
	n = 0
	for i in range(rows):
		for j in range(cols):
			im2var[i][j] = n
			n += 1
	A = np.zeros((2 * rows * cols + 1, rows * cols))
	b = np.zeros((2 * rows * cols + 1, 1))
	e = 0
	for y in range(rows):
		for x in range(cols-1):
			A[e][int(im2var[y][x+1])] = 1
			A[e][int(im2var[y][x])] = -1
			b[e] = im[y][x+1] - im[y][x]
			e += 1

	for y in range(rows - 1):
		for x in range(cols):
			A[e][int(im2var[y+1][x])] = 1
			A[e][int(im2var[y][x])] = -1
			b[e] = im[y+1][x] - im[y][x]
			e += 1 

	A[e][int(im2var[0][0])] = 1
	b[e] = im[0][0]
	A = sparse.csr_matrix(A)
	v = sparse.linalg.lsqr(A, b)[0]
	return np.reshape(v, (rows, cols))

def poisson(im1, im2, msk):
	# Essentially similar to the update_toy function, but with boundary checks and certain variables.
	rows = len(im1)
	cols = len(im1[0])
	im2var = np.zeros((rows, cols))
	n = 0
	for i in range(rows):
		for j in range(cols):
			if msk[i][j]:
				im2var[i][j] = n
				n += 1
			else:
				im2var[i][j] = -1

	msk_count = np.count_nonzero(msk)
	A = np.zeros((4 * msk_count, msk_count))
	b = np.zeros((4 * msk_count, 1))
	e = 0
	for y in range(rows):
		for x in range(cols):
			if msk[y][x]:
				if y+1 < rows and msk[y+1][x]:
					A[e][int(im2var[y+1][x])] = -1
					b[e] = im1[y][x] - im1[y+1][x]
				else:
					b[e] = im2[y][x]
				A[e][int(im2var[y][x])] = 1
				e += 1
				if y-1 >= 0 and msk[y-1][x]:
					A[e][int(im2var[y-1][x])] = -1
					b[e] = im1[y][x] - im1[y-1][x]
				else:
					b[e] = im2[y][x]
				A[e][int(im2var[y][x])] = 1
				e += 1
				if x+1 < cols and msk[y][x+1]:
					A[e][int(im2var[y][x+1])] = -1
					b[e] = im1[y][x] - im1[y][x+1]
				else:
					b[e] = im2[y][x]
				A[e][int(im2var[y][x])] = 1
				e += 1
				if x-1 >= 0 and msk[y][x-1]:
					A[e][int(im2var[y][x-1])] = -1
					b[e] = im1[y][x] - im1[y][x-1]
				else:
					b[e] = im2[y][x]
				A[e][int(im2var[y][x])] = 1
				e += 1

	A = sparse.csr_matrix(A)
	v = sparse.linalg.lsqr(A, b)[0]
	return v

def poisson_blend(im1, im2, msk):
	r1 = im1[:,:,0]
	g1 = im1[:,:,1]
	b1 = im1[:,:,2]
	r2 = im2[:,:,0]
	g2 = im2[:,:,1]
	b2 = im2[:,:,2]
	r_blend = poisson(r1, r2, msk)
	g_blend = poisson(g1, g2, msk)
	b_blend = poisson(b1, b2, msk)
	rows = len(im1)
	cols = len(im1[0])

	n = 0
	for y in range(rows):
		for x in range(cols):
			if msk[y][x]:
				r2[y][x] = r_blend[n]
				g2[y][x] = g_blend[n]
				b2[y][x] = b_blend[n]
				n += 1
	return np.dstack([r2, g2, b2])
