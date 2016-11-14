
import cv2
import numpy as np
from numpy import linalg as LA
import os

fold = 'GoodCells/ByProgram(not prepared)/'
folder = fold + 'TMP(1T)'
new_folder = fold + 'tmp(2)'

num = 0
part_size = 50

for file in os.listdir(folder):
	image_file = os.path.join(folder, file)
	img = cv2.imread(image_file)
	img = img[:,:,0]
	print(file)
	#print('Laplacian before:', cv2.Laplacian(img, cv2.CV_64F).var())
	num += 1
	x_size = img.shape[0]
	y_size = img.shape[1]
	
	"""for x in range(0, x_size//part_size):
		for y in range(0, y_size//part_size):
			#print('NUM', num)
			num+=1
			x2 = (x + 1) * part_size
			y2 = (y + 1) * part_size
			crop_img = img[x * part_size: x2, y * part_size: y2]"""
	for x in range(0, 4 * x_size//part_size):
		for y in range(0, 4 * y_size//part_size):
			num+=1
			x2 = (x//4 + 1) * part_size + (x % 4) * (part_size // 4)
			y2 = (y//4 + 1) * part_size + (y % 4) * (part_size // 4)
			crop_img = img[x2 - part_size: x2, y2 - part_size: y2]
			if crop_img.shape[0] < part_size or crop_img.shape[1] < part_size: 
				continue
				
			if cv2.Laplacian(crop_img, cv2.CV_64F).var() > 300:
				blank = np.zeros((part_size, part_size), np.uint8)
				av = crop_img.mean()
				print(av)
				for x in range(blank.shape[0]):
					for y in range(blank.shape[1]):
						blank[x,y] = int(abs(av - crop_img[x,y]))

				cv2.imwrite(new_folder + '/' + str(num) + '(' + str(int(cv2.Laplacian(crop_img, cv2.CV_64F).var())) + ', disp ' + str(int(LA.norm(blank,1)/(part_size*part_size//4))) + ')' +  '.tif', crop_img)

#"""
