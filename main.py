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

num_labels = 3#4
max_pictures_num = 350
max_training_size = 900

pixel_depth = 255.0
image_size = 100	# == PART SIZE

#pickle_file = 'data(100).pickle'

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
		#image = random.choice(os.listdir(folder))
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
	#print(np.mean(dataset))

	return dataset, labels, image_index

def load_dataset(folder, max_num, dataType):

	num1 = max_num;
	num2 = max_num;
	num3 = max_num;
	print(num1, 'num1')
	print(num2, 'num2')
	print(num3, 'num3')
	dataset_0, labels_0, image_index_0 = load_pictures(folder + '/1Taxol', num1, 0, dataType) 
	dataset_1, labels_1, image_index_1 = load_pictures(folder + '/01Taxol', num2, 1, dataType) 
	dataset_2, labels_2, image_index_2 = load_pictures(folder + '/Control', num3, 2, dataType) 
	dataset = np.concatenate((dataset_0, dataset_1, dataset_2), axis = 0)
	labels = np.concatenate((labels_0, labels_1, labels_2), axis = 0)
	print('Dataset: ', dataset.shape)
	
	return dataset, labels

def make_pickle(name):

	my_dict = {
		'Pickles/train_dataset': train_dataset,   
		'Pickles/train_labels': train_labels,
		'Pickles/valid_dataset': valid_dataset,
		'Pickles/valid_labels': valid_labels,
		'Pickles/test_dataset': test_dataset,
		'Pickles/test_labels': test_labels
		}
	try:
		#f = open(pickle_file, 'wb') 
		f = open(name, 'wb') 
		save = {
			name: my_dict[name]
			#'train_dataset': train_dataset,
			#'train_labels': train_labels,
			#'valid_dataset': valid_dataset,
			#'valid_labels': valid_labels,
			#'test_dataset': test_dataset,
			#'test_labels': test_labels
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save', ':', e)	
		raise



def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels



#"""

train_dataset, train_labels = load_dataset('Good archive', max_training_size, 0)
train_dataset, train_labels = randomize(train_dataset, train_labels)
print(train_dataset.shape)
print(train_labels.shape)

valid_dataset, valid_labels = load_dataset('Good archive', max_pictures_num, 1)              
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)			
print(valid_dataset.shape)								
print(valid_labels.shape)							
									
test_dataset, test_labels = load_dataset('Good archive', max_pictures_num, 2)
test_dataset, test_labels = randomize(test_dataset, test_labels)
print(test_dataset.shape)
print(test_labels.shape)

# make pickles
""" 
make_pickle('Pickles/train_dataset')
make_pickle('Pickles/train_labels')
make_pickle('Pickles/valid_dataset')
make_pickle('Pickles/valid_labels')
make_pickle('Pickles/test_dataset')
make_pickle('Pickles/test_labels')
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)"""

#"""

#open pickles

"""
with open('Pickles/train_dataset', 'rb') as f1:
	save1 = pickle.load(f1)
	train_dataset = save1['Pickles/train_dataset']
	del save1
with open('Pickles/train_labels', 'rb') as f2:
	save2 = pickle.load(f2)
	train_labels = save2['Pickles/train_labels']
	del save2

with open('Pickles/valid_dataset', 'rb') as f3:
	save3 = pickle.load(f3)
	valid_dataset = save3['Pickles/valid_dataset']
	del save3
with open('Pickles/valid_labels', 'rb') as f4:
	save4 = pickle.load(f4)
	valid_labels = save4['Pickles/valid_labels']
	del save4

with open('Pickles/test_dataset', 'rb') as f5:
	save5 = pickle.load(f5)
	test_dataset = save5['Pickles/test_dataset']
	del save5
with open('Pickles/test_labels', 'rb') as f6:
	save6 = pickle.load(f6)
	test_labels = save6['Pickles/test_labels']
	del save6


print(train_dataset.shape)
print(train_labels.shape)
print(valid_dataset.shape)
print(valid_labels.shape)
print(test_dataset.shape)
print(test_labels.shape) 	
"""
	
#check data
""" 
for i in range(5):
	crop = train_dataset[i, :, :]
	sm.imsave(str(i) + '_' + str(train_labels[i]) + 'tmp.tif', crop)

for i in range(5):
	crop = valid_dataset[i, :, :]
	sm.imsave(str(i) + '_' + str(valid_labels[i]) + 'tmp.tif', crop)
for i in range(5):
	crop = test_dataset[i, :, :]
	sm.imsave(str(i) + '_' + str(test_labels[i]) + 'tmp.tif', crop)"""

#find overlaps
"""
def overlap(dataset1, dataset2):
	num = 0;
	for i, img1 in enumerate(dataset1):
		for j, img2 in enumerate(dataset2):
			if np.array_equal(img1, img2):
				#print(i, j)
				num += 1
	print(num)
print('First overlap:')
overlap(test_dataset, train_dataset)
print('Second overlap:')
overlap(valid_dataset, train_dataset)
print('Third overlap:')
overlap(test_dataset, valid_dataset)

"""

#start of learning

#"""

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


print('Training set: ', train_dataset.shape, train_labels.shape)
print('Valid set: ', valid_dataset.shape, valid_labels.shape)
print('Test set: ', test_dataset.shape, test_labels.shape)

batch_size = 64#128
num_hidden_nodes = 128#128#512

graph = tf.Graph()
with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	
	global_step = tf.Variable(0)

	weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
	biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
	weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]))
	biases2 = tf.Variable(tf.zeros([num_labels]))

	lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
	logits = tf.matmul(lay1_train, weights2) + biases2
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	#learning_rate = tf.train.exponential_decay(0.02, global_step, 10000, 0.95, staircase=True)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)	
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
	valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
	lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
	test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

num_steps = 5001
																							

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
def class_accuracy(predictions, labels, class_num):
	num = 0;
	for i in range(0, predictions.shape[0]):		
		if (np.argmax(predictions[i, :]) == np.argmax(labels[i, :]) and np.argmax(predictions[i, :]) == class_num):
			num += 1
	return (100.0 * num / predictions.shape[0])


with tf.Session(graph=graph) as session:

	tf.initialize_all_variables().run()
	print('Initialized')
	for step in range(num_steps):

		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
		if (step % 100 == 0):
			print('Miniloss at step %d: %f' % (step, l))
			print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels));

			print(class_accuracy(predictions, batch_labels, 0), class_accuracy(predictions, batch_labels, 1), class_accuracy(predictions, batch_labels, 2))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))	
	print(class_accuracy(test_prediction.eval(), test_labels, 0), class_accuracy(test_prediction.eval(), test_labels, 1), 
	class_accuracy(test_prediction.eval(), test_labels, 2))
	
#"""


