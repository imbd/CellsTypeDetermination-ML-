import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sm
import os

folder = "01Taxol/Pos0/"
new_folder = "01Taxol/RightNames/"

for file in os.listdir(folder):
	new_file_name = file.split('_')[1]
	i = 0
	i = int(new_file_name)
	print(new_file_name, i)
	Img = plt.imread(os.path.join(folder, file))
	sm.imsave(new_folder + str(i) + '.tif', Img)
	
