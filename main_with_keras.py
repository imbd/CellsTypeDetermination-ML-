import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc as sm
import sys
import os
import random
import hashlib
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout,Activation
from keras.optimizers import SGD, Adam, RMSprop

seed = 7
np.random.seed(seed)


num_labels = 3
max_pictures_num = 300
max_training_size = 1000

pixel_depth = 255.0
image_size = 56#100

image_files = max_pictures_num


def load_pictures(folder, max_num, kind, dataType):
		
	dataset = np.ndarray(shape=(max_num, image_size, image_size), dtype = np.float32)
	labels = np.ndarray(max_num, dtype = np.int32)
	image_index = 0
	print(folder)
	print(max_num)
	tmp = 0
	t = 0
	for image in os.listdir(folder):

		t += 1	
		if (dataType == 1 and t <= max_training_size):
			continue 
		if (dataType == 2 and t <= max_training_size + max_pictures_num):
			continue 

		image_file = os.path.join(folder, image)
		Img = plt.imread(image_file)

		image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

		dataset[image_index, :, :] = image_data[:,:]#,0]
		labels[image_index] = kind
		image_index += 1
		if image_index == max_num:
			break			
	
	num_images = image_index
	dataset = dataset[0:num_images, :, :]
	labels = labels[0:num_images]
	
	print('datasetType', dataType)
	print(dataset.shape)
	print(labels.shape)

	return dataset, labels, image_index

def load_dataset(folder, max_num, dataType):

	num1 = max_num;
	num2 = max_num;
	num3 = max_num;
	print(num1, 'num1')
	print(num2, 'num2')
	print(num3, 'num3')
	dataset_0, labels_0, image_index_0 = load_pictures(folder + '1Taxol(56)', num1, 0, dataType) 
	dataset_1, labels_1, image_index_1 = load_pictures(folder + '01Taxol(56)', num2, 1, dataType) 
	dataset_2, labels_2, image_index_2 = load_pictures(folder + 'Control(56)', num3, 2, dataType) 
	dataset = np.concatenate((dataset_0, dataset_1, dataset_2), axis = 0)
	labels = np.concatenate((labels_0, labels_1, labels_2), axis = 0)
	print('Dataset: ', dataset.shape)
	
	return dataset, labels


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

train_dataset, train_labels = load_dataset('Good archive/', max_training_size, 0)
train_dataset, train_labels = randomize(train_dataset, train_labels)
print(train_dataset.shape)
print(train_labels.shape)

test_dataset, test_labels = load_dataset('Good archive/', max_pictures_num, 1)
test_dataset, test_labels = randomize(test_dataset, test_labels)
print(test_dataset.shape)

print(test_labels.shape)
valid_dataset, valid_labels = load_dataset('Good archive/', max_pictures_num, 2)              
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)			
print(valid_dataset.shape)								
print(valid_labels.shape)							
			


									
#65% conv model
#"""

train_dataset = train_dataset.reshape(train_dataset.shape[0], image_size, image_size,1)
test_dataset = test_dataset.reshape(test_dataset.shape[0], image_size, image_size,1)
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0], image_size, image_size,1)
train_dataset = train_dataset.astype('float32')
test_dataset = test_dataset.astype('float32')
valid_dataset = valid_dataset.astype('float32')


train_labels = np_utils.to_categorical(train_labels, num_labels)
test_labels = np_utils.to_categorical(test_labels, num_labels)
valid_labels = np_utils.to_categorical(valid_labels, num_labels)

pool_size = (3,3)  
kernel_size = (3,3)

model = Sequential()

model.add(Convolution2D(4, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=(image_size, image_size,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_labels, activation='softmax'))

#sgd = SGD(lr=1e-9)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(train_dataset, train_labels, nb_epoch=125, batch_size=32, verbose=2, validation_data=(valid_dataset, valid_labels))

scores= model.evaluate(test_dataset, test_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Test score:', scores[0])
print('Test accuracy:', scores[1])

#model.save('my_conv_new_tmp_model.h5')

#"""


# For loading
"""

model = load_model('my_conv_model.h5')
#scores= model.evaluate(test_dataset, test_labels)
scores= model.evaluate(valid_dataset, valid_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Test score:', scores[0])
print('Test aqccuracy:', scores[1])

"""


# Good model without convolution 
"""

train_dataset = train_dataset.reshape(train_dataset.shape[0], image_size*image_size)
test_dataset = test_dataset.reshape(test_dataset.shape[0], image_size*image_size)
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0], image_size*image_size)
train_dataset = train_dataset.astype('float32')
test_dataset = test_dataset.astype('float32')
valid_dataset = valid_dataset.astype('float32')

train_labels = np_utils.to_categorical(train_labels, num_labels)
test_labels = np_utils.to_categorical(test_labels, num_labels)
valid_labels = np_utils.to_categorical(valid_labels, num_labels)

model = Sequential()

model.add(Dense(256, input_shape=(image_size*image_size,),activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_labels, activation='softmax'))

#sgd = SGD(lr=1e-9)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(train_dataset, train_labels, nb_epoch=125, batch_size=32, verbose=2, validation_data=(valid_dataset, valid_labels))

scores= model.evaluate(test_dataset, test_labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Test score:', scores[0])
print('Test accuracy:', scores[1])

"""







