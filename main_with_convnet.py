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


num_labels = 2
max_pictures_num = 400
max_training_size = 1200

pixel_depth = 255.0
image_size = 28#100

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
	dataset_0, labels_0, image_index_0 = load_pictures(folder + '1Taxol(28)', num1, 0, dataType) 
	dataset_1, labels_1, image_index_1 = load_pictures(folder + '01Taxol(28)', num2, 1, dataType) 
	#dataset_2, labels_2, image_index_2 = load_pictures(folder + '/Control(28)', num3, 2, dataType) 
	dataset = np.concatenate((dataset_0, dataset_1), axis = 0)
	labels = np.concatenate((labels_0, labels_1), axis = 0)
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

valid_dataset, valid_labels = load_dataset('Good archive/', max_pictures_num, 1)              
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)			
print(valid_dataset.shape)								
print(valid_labels.shape)							
									
test_dataset, test_labels = load_dataset('Good archive/', max_pictures_num, 2)
test_dataset, test_labels = randomize(test_dataset, test_labels)
print(test_dataset.shape)
print(test_labels.shape)




			#CONVOLUTIONAL

num_channels = 1 # grayscale

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
	layer1_biases = tf.Variable(tf.zeros([depth]))
	layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
	layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
	layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	def model(data):
		conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + layer1_biases)
		conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
		hidden = tf.nn.relu(conv + layer2_biases)
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		return tf.matmul(hidden, layer4_weights) + layer4_biases
  
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
	optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
  
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 5001

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 100 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))










