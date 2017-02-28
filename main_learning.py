import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sm
import sys
import os
import os.path
from pathlib import Path
import random
import hashlib
from scipy import ndimage
import shutil
from keras.utils import np_utils
from keras.models import load_model

seed = 37
np.random.seed(seed)


learn_type = 0
TRAIN_NUMBER = 25 # can be changed
GUESS_NUMBER = 20 # can be changed
pixel_depth = 255.0
image_size = 50
num_labels = 3
data_folder = '_sources/Data/'
folders = ['1Taxol/', '01Taxol/', 'Control/']	
models_folder = '_sources/Models/'
model_name = ['main.h5', '01t_1t.h5', 'c_1t.h5']
train_folder = 'Train/'
guess_folder = 'Guess/'

print()
print("Print wanted type of classifying(\"exit\" for exit)")
print("0 - all 3 types")
print("1 - 1Taxol and 01Taxol")
print("2 - 1Taxol and Control")
inputLine = sys.stdin.readline()

exit = False
while (True):

	if (inputLine[:-1] == '0'):
		break
	if (inputLine[:-1] == '1'):
		learn_type = 1
		break
	if (inputLine[:-1] == '2'):
		learn_type = 2
		break
	if (inputLine[:-1].lower() == "exit".lower()):
		exit = True
		break	
	inputLine = sys.stdin.readline()

if exit:
	sys.exit()

if learn_type != 0:
	num_labels = 2
if learn_type == 2:
	folders =  ['1Taxol/','Control/', '01Taxol/']

dataset = np.ndarray(shape=(num_labels * GUESS_NUMBER, image_size, image_size), dtype = np.float32)
labels = np.ndarray(num_labels * GUESS_NUMBER, dtype = np.int32)

	

def load(dest_folder,number, load_type):

	print('Loading data')
	if os.path.exists(dest_folder):
		shutil.rmtree(dest_folder)
	for f in folders:
		num = 0
		folder = data_folder + f
		if (learn_type != 0 and f == folders[2]):
			continue
		os.makedirs(dest_folder+f)
		k = number
		while (k > 0):
			file = random.choice(os.listdir(folder))
			image_file = os.path.join(folder, file) 
			Img = plt.imread(image_file)
			my_file = Path(dest_folder + str(hash(file) % (10 ** 8)) +"_" + f[0:-1] + ".tif")
			if (not my_file.is_file()):
				if load_type == 1:
					num += 1
					sm.imsave(dest_folder + f + str(num) + '.tif', Img)
				else:
					sm.imsave(dest_folder + str(hash(file) % (10 ** 8)) +"_" + f[0:-1] + '.tif', Img)
				k -= 1
			
	if (load_type == 2):
		num = 0
		inn = 0
		for filename in os.listdir(dest_folder):
			if (filename.endswith(".tif")):
				im_class = filename[filename.index('_')+1:-4]
				im_class_num = 0
				if (im_class == folders[1][0:-1]):
					im_class_num = 1
				if (im_class == folders[2][0:-1]):
					im_class_num = 2				
				if (im_class_num < num_labels):
					labels[num] = im_class_num
					for file in os.listdir(data_folder + folders[im_class_num][0:-1] +'(small)'):
						if (str(hash(file[2:len(file)])% (10 ** 8)) == filename[0: filename.index('_')]):
							inn += 1
							small_file = os.path.join(data_folder + folders[im_class_num][0:-1] +'(small)',file)
							image_data = (ndimage.imread(small_file).astype(float) - pixel_depth / 2) / pixel_depth
							dataset[num,:,:] =  image_data[:,:]
					os.rename(dest_folder + filename, dest_folder + str(num) + ".tif")
					num += 1
		for filename in os.listdir(dest_folder):
			if os.path.isfile(dest_folder + filename):
				n = filename.find('_')
				if (n != -1):
					os.remove(os.path.join(dest_folder, filename))
			if os.path.isdir(dest_folder + filename):
				if learn_type != 0 and filename == folders[2][:-1]:
					shutil.rmtree(dest_folder + filename) 

load(train_folder, TRAIN_NUMBER, 1)	
load(guess_folder, GUESS_NUMBER, 2)

dataset = dataset.reshape(dataset.shape[0], image_size, image_size,1)
dataset = dataset.astype('float32')

old_labels = np.copy(labels)
labels = np_utils.to_categorical(labels, num_labels)

for i in range (dataset.shape[0]):
	for j in range (i + 1, dataset.shape[0]):
		if (dataset[i] == dataset[j]).all():
			print("Equal: ", i, j)		


model = load_model(models_folder + model_name[learn_type])
print("Model: ", model_name[learn_type])
scores= model.evaluate(dataset, labels, verbose=0)

print('Test accuracy:', str(100*round(scores[1],3)) + '%')

print("You can now train before your own classification. For that, look at pictures in 'Train' folder.")
print("After go to 'Guess' folder and spread all pictures into folders. Print \"Yes\" when you are ready(\"exit\" for exit)")
inputLine = sys.stdin.readline()

exit = False
while (not inputLine[:-1].lower() == "Yes".lower()):
	if (inputLine[:-1].lower() == "exit".lower()):
		exit = True
		break	
	inputLine = sys.stdin.readline()

if exit:
	sys.exit()


print("Start of checking results!")	
right_number = 0
wrong_number = 0
for f in folders:
	folder = guess_folder + f
	if not os.path.exists(folder):
		continue
	for file in os.listdir(folder):
		num = int(file[:-4])
		if (folders[old_labels[num]] == f):
			right_number += 1
		else:
			wrong_number += 1
			print('You did a mistake: picture ' + str(num) + '.tif is from ' + folders[old_labels[num]][0:-1] + ', not from ' + f[0:-1])
		
print('Classified pictures number:', right_number + wrong_number)
if (right_number + wrong_number > 0):
	print('Your classifying accuracy:' + str(100*round(right_number / (right_number + wrong_number),3)) + '%')






