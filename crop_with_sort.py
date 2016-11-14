
import cv2
import numpy as np
from numpy import linalg as LA
import os

fold = 'All data/'
folder = fold + 'ByProgram/Second/C'
new_folder = fold + 'tmp(C)'

num = 0
part_size = 100

for file in os.listdir(folder):
	image_file = os.path.join(folder, file)
	img = cv2.imread(image_file)
	img = img[:,:,0]
	print(file)
	num += 1
	x_size = img.shape[0]
	y_size = img.shape[1]

	best_img = img[0:part_size, 0:part_size]
	
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
			if crop_img.mean() > best_img.mean(): #or (cv2.Laplacian(best_img, cv2.CV_64F).var() < 200 and cv2.Laplacian(crop_img, cv2.CV_64F).var() > 200):
				best_img = img[x2 - part_size: x2, y2 - part_size: y2]
			
	#if cv2.Laplacian(best_img, cv2.CV_64F).var() > 200:	
	cv2.imwrite(new_folder + '/' + '_'  + str(num) + '(' + str(int(cv2.Laplacian(best_img, cv2.CV_64F).var())) +  ')' +  '.tif', best_img)		
	print('Laplacian:', cv2.Laplacian(best_img, cv2.CV_64F).var())

#"""









