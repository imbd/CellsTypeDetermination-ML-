import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import scipy.misc as sm
import os

fold = 'Good archive/old/'
folder = fold + 'c'
num = 0
for file in os.listdir(folder):
	image_file = os.path.join(folder, file)
	Img = plt.imread(image_file)
	if num % 100 == 0: 
		print(num)
	num += 1
	x = np.size(Img, 0)
	y = np.size(Img, 1)
	crop_img = Img[0:x, 0:y]
	tmp = Img[0:x, 0:y].copy()

	for i in range(x):
		for j in range(y):
			tmp[i,j] = crop_img[x-1-i,y-1-j] 
		
	sm.imsave(folder + '/' + str(num) + '.tif', tmp)
	num += 1
	for i in range(x):
		for j in range(y):
			tmp[i,j] = crop_img[i,y-1-j] 
		
	sm.imsave(folder +  '/' + str(num) + '.tif', tmp)
	num += 1
	for i in range(x):
		for j in range(y):
			tmp[i,j] = crop_img[x-1-i,j] 
		
	sm.imsave(folder +  '/' + str(num) + '.tif', tmp)
	num += 1
	sm.imsave(folder + '/' + str(num) + '.tif', np.rot90(tmp,1))
	num += 1
	sm.imsave(folder + '/' + str(num) + '.tif', np.rot90(tmp,3))
	num += 1

