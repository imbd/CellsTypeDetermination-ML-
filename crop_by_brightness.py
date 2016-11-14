import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import scipy.misc as sm
import os
from os import path
from pathlib import Path
import math

fold = '01Taxol'
#fold = 'Control'
folder = fold + '/Pos0'
new_folder = fold + '/Pieces'
#new_folder = 'Control(pieces)'
files = os.listdir(folder)
my_path = Path(folder) 
num = 0
pic = 0

for file in files:	
	if not os.path.isfile(os.path.join(os.getcwd(),folder + '/' + file)):
		continue	
	Img = plt.imread(folder + '/' + file)
	pic += 1
	x_size = np.size(Img, 0)
	y_size = np.size(Img, 1)
	part_size = 100
	#avg = 40
	l = []
	delete = []
	print(file)	
	for x in range(1, x_size//part_size - 1): 
		for y in range(1, y_size//part_size - 1):
			num+=1
			x2 = (x + 1) * part_size
			y2 = (y + 1) * part_size
			crop_img = Img[x * part_size: x2, y * part_size: y2]
			max_x = crop_img.argmax() // part_size
			max_y = crop_img.argmax() - (part_size * max_x)		
			if crop_img.max() > 0: #and np.average(crop_img) > 30 and np.average(crop_img) < 90: 
				l.append((x * part_size + max_x, y* part_size + max_y))
				delete.append(False)
	
	l.sort(key=lambda x: -Img[x[0],x[1]])
	print(len(l))
	
	for i in range(len(l)):
		for j in range(i + 1, len(l)):
			if delete[i] == True or delete[j] == True:
				continue
			x1 = l[i][0]
			y1 = l[i][1]
			x2 = l[j][0]
			y2 = l[j][1]
			if math.sqrt((x1-x2)**2 + (y1-y2)**2) < part_size: 
				delete[j] = True
			
	
	l_num = 0
	ind = -1
	for el in l:
		l_num += 1
		ind += 1
		if delete[ind] == True:
			l_num -= 1
			continue
		print(el[0], el[1], Img[el[0],el[1]])
		crop_img = Img[el[0] - part_size: el[0] + part_size, el[1] - part_size: el[1] + part_size]
		file_name = new_folder + '/' + str(pic) + ',' + str(l_num) + '(' + 'br=' + str(crop_img.max()) + ',' + 'av=' + str(int(np.average(crop_img))) + ',' + 'mn=' + str(int(np.linalg.norm(crop_img))) +')'+'.tif'
		if np.linalg.norm(crop_img) > 15000 or np.average(crop_img) < 35 or crop_img.max() < 150:
			l_num -= 1
			continue
		sm.imsave(file_name, crop_img)
		if l_num >= 10:			#maybe not needed
			break

