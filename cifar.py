import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color

import pickle
import os

class Cifar():

	def one_hot(labels, size):
		"""
		Create one-hot encodings for each of the class labels
		"""
		a = np.zeros((len(labels), size), 'uint8')
		for ind in range(len(labels)):
			a[ind][labels[ind]] = 1
		return a

	def unpickle(file):
		"""
		Unpickle the CIFAR-10 file
		"""
		fo = open(file, 'rb')
		dict = pickle.load(fo, encoding='bytes')
		fo.close()
		return dict

	def _convert_images(raw, num_channels, dim_size):
		"""
		Convert images from the CIFAR-10 format and
		return a 4-dim array with shape: [image_number, height, width, channel]
		where the pixels are floats between 0.0 and 1.0.
		"""

		# Convert the raw images from the data-files to floating-points.
		raw_float = np.array(raw, dtype=float) / 255.0

		# Reshape the array to 4-dimensions.
		images = raw_float.reshape([-1, num_channels, dim_size, dim_size])

		# Reorder the indices of the array.
		images = images.transpose([0, 2, 3, 1])
		return images

	def __init__(self, path):
		train_data_batches = [Cifar.unpickle(path +'/data_batch_'+str(i)) for i in range(1, 6)]
		test_data_batch = Cifar.unpickle(path +'/test_batch')

		self.val_images = color.rgb2lab(Cifar._convert_images(train_data_batches[0][b'data'], 3, 32))
		self.val_labels = np.array(train_data_batches[0][b'labels'])

		self.train_images = color.rgb2lab(Cifar._convert_images(train_data_batches[1][b'data'], 3, 32))
		self.train_labels = np.array(train_data_batches[1][b'labels'])
		for i in range(2, 5):
			self.train_images = np.append(self.train_images, color.rgb2lab(Cifar._convert_images(train_data_batches[i][b'data'], 3, 32)), axis=0)
			self.train_labels = np.append(self.train_labels, np.array(train_data_batches[i][b'labels']), axis=0)

		self.test_images = color.rgb2lab(Cifar._convert_images(test_data_batch[b'data'], 3, 32))
		self.test_labels = np.array(test_data_batch[b'labels'])

		print(self.test_images.shape)

	def data(self, batch_size, is_supervised):
	    if is_supervised:
	        for _ in range(int(len(self.train_images)/batch_size)):
	            indices = np.random.randint(0, len(self.train_images), size=batch_size)
	            x = self.train_images[indices]
	            y = self.train_labels[indices]
	            y = Cifar.one_hot(y, 10)
	            plt.imshow(np.squeeze(x[0]))
	            plt.show()
	            yield x, y

	    else:
	    	for _ in range(int(len(self.train_images)/batch_size)):
	    		indices = np.random.randint(0, len(self.train_images), size=batch_size)
	    		x = self.train_images[indices]
	    		plt.imshow(np.squeeze(x[0]))
	    		yield x
