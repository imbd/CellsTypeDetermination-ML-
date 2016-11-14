import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import scipy.misc as sm
import os
from os import path
from pathlib import Path
import math
from PIL import Image

fold = 'GoodCells/'
folder = fold + 'TMP'
new_folder = fold + 'TMP_br'

files = os.listdir(folder)
BR = 50
num = 0

for file in files:	
	if not os.path.isfile(os.path.join(os.getcwd(),folder + '/' + file)):
		continue	
	img = plt.imread(folder + '/' + file)
	x = np.size(img, 0)
	y = np.size(img, 1)
	k = BR / np.average(img)
	def f(x):
		if x > 0: 
			if (255/k < x - 1):
				return 255 
		return ((int) (max(min((k * x), 255), 0)))
	
	f = np.vectorize(f)
	arr = f(img)

	def g(x):
		return (255 -  x)
	
	g = np.vectorize(g)
	arr = g(arr)

	print(np.average(img))
	print(np.average(arr))
	print(arr)
	print(img.max())
	print(img.min())
	print(arr.max())
	print(arr.min())
	print(arr.shape)
	plt.imsave(new_folder + '/' + str(num), img)
	plt.imsave(new_folder + '/' + str(num) + '(' + str(int(np.average(arr))) + ')', arr)

	num += 1
