# mnist_tf second version

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import numpy as np
import matplotlib.pyplot as plt

# Train images size
def train_size(num):
	print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
	print ('--------------------------------------------------')
	x_train = mnist.train.images[:num,:]
	print ('x_train Examples Loaded = ' + str(x_train.shape))
	y_train = mnist.train.labels[:num,:]
	print ('y_train Examples Loaded = ' + str(y_train.shape))
	print('')
	return x_train, y_train

# Test image size
def test_size(num):
	print ('Total Test Images in Dataset = ' + str(mnist.test.images.shape))
	print ('--------------------------------------------------')
	x_test = mnist.test.images[:num,:]
	print ('x_test Examples Loaded = ' + str(x_test.shape))
	y_test = mnist.test.labels[:num,:]
	print ('y_test Examples Loaded = ' + str(y_test.shape))
	print('')
	return x_test, y_test

# Functions to resize and display the data
def display_digit(num):
	print(y_train[num])
	label = y_train[num].argmax(axis=0)
	image = x_train[num].reshape([28,28])
	plt.title('Example: %d Label: %d' % (num, label))
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))
	plt.show()


# def display_mult_flat(start, stop):
#     images = x_train[start].reshape([1,784])
#     for i in range(start+1,stop):
#         images = np.concatenate((images, x_train[i].reshape([1,784])))
#     plt.imshow(images, cmap=plt.get_cmap('gray_r'))
#     plt.show()

x_train, y_train = train_size(55000)
# display_digit(np.random.randint(0, x_train.shape[0]))

import tensorflow as tf
sess = tf.Session()

# Placeholder to feed x_train
x = tf.placeholder(tf.float32, shape[None, 784])

# Plceholder to feed y_train
y_ = tf.placeholder(tf.float32, shape[None, 10])

	