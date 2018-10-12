#main.py
# CS194-26 (CS294-26): Project 3

# Jacky Tian

from hybrid_image import *
from proj3 import *

### Part 1.1
kernel = np.array([[1.0, 4, 7, 4, 1],
				  [4, 16, 26, 16, 4],
				  [7, 26, 41, 26, 7],
				  [4, 16, 26, 16, 4],
				  [1, 4, 7, 4, 1]])
for i in range(len(kernel)):
	for j in range(len(kernel[0])):
		kernel[i][j] = float(kernel[i][j])/273.0

im_out = sharpen('kobe.jpg', kernel)
plt.imsave('kobe_sharp.jpg', im_out)

# # ### Part 1.2
sigma1 = 4
sigma2 = 8
for pair in [('old.jpg', 'young.jpg'), ('horse.jpeg', 'zebra.jpg'), ('sad.jpeg', 'happy.jpg')]:
	# high sf
	im1 = plt.imread(pair[0])/255.

	# low sf
	im2 = plt.imread(pair[1])/255

	im1_aligned, im2_aligned = align_images(im2, im1)
	hybrid, high, low = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
	fname = pair[0][:-4] + pair[1][:-3] + 'jpg'
	plt.imsave(fname, hybrid, cmap='gray')
	if pair[0] == 'old.jpg':
		old_ft = rescale_intensity(fourier(im1))
		plt.imsave('old_ft.jpg', old_ft, cmap='gray')
		young_ft = rescale_intensity(fourier(im2))
		plt.imsave('young_ft.jpg', young_ft, cmap='gray')
		high_ft = rescale_intensity(fourier(high))
		plt.imsave('high_ft.jpg', high_ft, cmap='gray')
		low_ft = rescale_intensity(fourier(low))
		plt.imsave('low_ft.jpg', low_ft, cmap='gray')
		hybrid_ft = rescale_intensity(fourier(hybrid))
		plt.imsave('hybrid_ft.jpg', hybrid_ft, cmap='gray')

# ### Part 1.3
im = plt.imread('monalisa.jpg')/255.
gauss = gaussian_stack(im)
lap = laplacian_stack(im, gauss)
both = np.concatenate([gauss, lap])
fig, plots = plt.subplots(nrows=2, ncols=len(gauss))
fig.set_size_inches(20, 10)
i = 0
for plot in plots.flat:
	plot.imshow(rescale_intensity(both[i], in_range=(-0.5, 0.5), out_range=(0, 1)))
	plot.axis('off')
	i += 1

plt.savefig('monalisa_stack.jpg')

### Part 1.4
for pair in [('sad.jpeg', 'trump.jpg'), ('apple.jpeg', 'old.jpg')]:
	im1 = plt.imread(pair[0])/255.
	im2 = plt.imread(pair[1])/255.
	if pair[0] == 'apple.jpeg':
		im2 = im2[400:700, 300:600]
		msk = np.ones((300, 300))
		msk[50:100, :120] = 0
		msk[50:100, 190:] = 0
		msk[235:, 50:250] = 0
	else:
		im2 = im2[:265, 120:310]
		im1, im2 = align_images(im1, im2)
		msk = np.hstack([np.ones((int(im1.shape[0]), int(np.floor(im1.shape[1]/2)))), np.zeros((int(im1.shape[0]), int(np.ceil(im1.shape[1]/2))))])
	m = blend(im1, im2, msk)
	plt.imsave(pair[0][:-4]+pair[1][:-4]+'.jpg', m, cmap=None)

# ## Part 2.1
im_toy = plt.imread('toy.png')/255.
new_toy = update_toy(im_toy)
plt.imsave('new_toy.jpg', new_toy, cmap='gray')

### Part 2.2
source = plt.imread('penguin.jpg')/255.
target = plt.imread('im3.jpg')/255.
#s = np.zeros((len(target), len(target[0]), 3))
for y in range(1700, 2000):
	for x in range(500, 833):
		target[y][x] = source[y-1500][x-500]
m = np.zeros((len(target), len(target[0])))
m[1500:2000, 500:833] = 1
#im_blend = poisson_blend(s, target, m)
plt.imsave('blend.jpg', target)
