# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
from scipy.misc import imresize
 
def run(files, contrast=False):
	# auto contrast channels by manipulating the intensities
	def auto_contrast_high(img):
		ret_img = []
		for i in img:
			temp = []
			for j in i:
				if j > 0.9: 
					temp.append(1)
				else:
					temp.append(j)
			ret_img.append(temp)
		return np.array(ret_img)

	def auto_contrast_low(img):
		ret_img = []
		for i in img:
			temp = []
			for j in i:
				if j < 0.1:
					temp.append(0)
				else:
					temp.append(j)
			ret_img.append(temp)
		return np.array(ret_img)

	def contrast(b, g, r):
		max_sum = b.sum().sum()
		min_sum = b.sum().sum()
		max_img = 0
		min_img = 0
		sums = [g.sum().sum(), r.sum().sum()]
		for i in range(len(sums)):
			if sums[i] > max_sum:
				max_sum = sums[i]
				if i == 0:
					max_img = 1
				else:
					max_img = 1
			else:
				min_sum = sums[i]
				if i == 0:
					min_img = 2
				else:
					min_img = 2
		colors = [b, g, r]
		colors[max_img] = auto_contrast_high(colors[max_img])
		colors[min_img] = auto_contrast_low(colors[min_img])
		return colors[0], colors[1], colors[2]

	# naive function for finding the best alignment of two images
	def ncc(img1, img2):
		return ((img1/np.linalg.norm(img1)) * (img2/np.linalg.norm(img2))).ravel().sum()

	# image pyramid function for handling larger pictures
	def pyramid(img1, img2):
		i = len(img1)
		j = len(img1[0])
		if max(i, j) > 200:
			new_img1 = imresize(img1, 0.5)
			new_img2 = imresize(img2, 0.5)
			scaled = pyramid(new_img1, new_img2)
			shift = [x*2 for x in scaled] 
		else:
			shift = find_roll(img1, img2)
		return shift

	# finds the best roll for aligning images
	def find_roll(img1, img2):
		border = len(img2)//10
		section2 = img2[border:len(img2)-border, border:len(img2)-border]
		bestscore = 0
		best_i, best_j = 0, 0
		for i in range(30):
			for j in range(30):
				new_img1 = np.roll(img1, i, 0)
				new_img1 = np.roll(new_img1, j, 1)
				section1 = new_img1[border:len(img1)-border, border:len(img1)-border]
				score = ncc(section1, section2)
				if score > bestscore:
					bestscore = score
					best_i = i
					best_j = j
		return [best_i, best_j]

	# align the images
	# functions that might be useful for aligning the images include:
	# np.roll, np.sum, sk.transform.rescale (for multiscale)
	def align(img1, img2):
		if max(len(img1), len(img1[0])) > 400:
			best = pyramid(img1, img2)
		else:
			best = find_roll(img1, img2)
		ret_img1 = np.roll(img1, best[0], 0)
		ret_img1 = np.roll(ret_img1, best[1], 1)
		return ret_img1

	for imname in files:
		# read in the image
		im = skio.imread(imname)

		# convert to double (might want to do this later on to save memory)    
		im = sk.img_as_float(im)
		    
		# compute the height of each part (just 1/3 of total)
		height = np.floor(im.shape[0] / 3.0).astype(np.int)

		# separate color channels
		b = im[:height]
		g = im[height: 2*height]
		r = im[2*height: 3*height]

		if contrast:
			b, g, r = contrast(b, g, r)

		if imname == 'emir.tif' or imname == 'self_portrait.tif' or imname == 'monastery.jpg':
			ab = align(b, g)
			ar = align(r, g)
			# create a color image
			im_out = np.dstack([ar, g, ab])
		else:
			ag = align(g, b)
			ar = align(r, b)
			im_out = np.dstack([ar, ag, b])

		# save the image
		fname = 'out_path/' + imname[:-3] + 'jpg'
		skio.imsave(fname, im_out)

		# display the image
		# skio.imshow(im_out)
		# skio.show()
