import cv2 
import numpy as np
from numpy import linalg as LA
import os


fold = 'Good archive/'
folder = fold + '01Taxol/'
new_folder = fold + '01Taxol(100)/'
part_size = 100

for file in os.listdir(folder):
	image_file = os.path.join(folder, file)
	img = cv2.imread(image_file)
	res = cv2.resize(img[:,:,0], (100, 100)) 
	print(cv2.mean(res))
	cv2.imwrite(new_folder + '/' + file, res)

