import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import scipy.misc as sm
import os

def crop(file_name, directory):

	part_size = 50
	FILE = file_name
	directory = FILE.split('.')[0]
	print(directory)

	Img2 = plt.imread(FILE)

	if not os.path.exists(directory):
		os.makedirs(directory)
		print("YEP")

	x_size = np.size(Img2, 0)
	y_size = np.size(Img2, 1)

	print(Img2)
	print(Img2.shape)
	print(Img2.dtype)
	print(Img2.min())
	print(Img2.max())
	print(np.average(Img2))
	avg = 60
	num = 0

	for x in range(0, x_size//part_size):
		for y in range(0, y_size//part_size):
			file_name = directory + '/' + directory + '_' + str(num) + '.tif'
			num+=1

			x2 = (x + 1) * part_size
			y2 = (y + 1) * part_size
			crop_img = Img2[x * part_size: x2, y * part_size: y2, 0]
			print(np.average(crop_img))		
			if (np.average(crop_img) > avg):
				sm.imsave(file_name, crop_img)

