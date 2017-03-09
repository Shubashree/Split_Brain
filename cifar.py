import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import color

import pickle
import os

np.random.seed(31415)

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

        self.val_images = Cifar._convert_images(train_data_batches[0][b'data'], 3, 32)
        self.val_labels = np.array(train_data_batches[0][b'labels'])

        self.train_images = Cifar._convert_images(train_data_batches[1][b'data'], 3, 32)
        self.train_labels = np.array(train_data_batches[1][b'labels'])
        for i in range(2, 5):
            self.train_images = np.append(self.train_images, Cifar._convert_images(train_data_batches[i][b'data'], 3, 32), axis=0)
            self.train_labels = np.append(self.train_labels, np.array(train_data_batches[i][b'labels']), axis=0)

        self.test_images = Cifar._convert_images(test_data_batch[b'data'], 3, 32)
        self.test_labels = np.array(test_data_batch[b'labels'])

        #print(self.test_images.shape)

    def normalize(lab_images):
        lab_images[:, :, :, 0] = lab_images[:, :, :, 0] / 100.0
        lab_images[:, :, :, 1] = lab_images[:, :, :, 1] / 99.0
        lab_images[:, :, :, 2] = lab_images[:, :, :, 2] / 108.0
        return lab_images

    def denormalize_image(lab_image):
        lab_image[:, :, 0] = lab_image[:, :, 0] * 100.0
        lab_image[:, :, 1] = lab_image[:, :, 1] * 99.0
        lab_image[:, :, 2] = lab_image[:, :, 2] * 108.0
        return lab_image

    def quantize(lab_images):
        lab_images[:, :, :, 0] = np.digitize(lab_images[:, :, :, 0], np.linspace(0, 101, 100))
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(-88, 99, 200))
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(-108, 95, 200))
        return lab_images

    def convert_to_lab(self):
        self.val_images = color.rgb2lab(self.val_images)
        self.train_images = color.rgb2lab(self.train_images)
        np.random.shuffle(self.train_images)
        self.test_images = color.rgb2lab(self.test_images)
        #print(self.val_images[0])

        # self.quantized_val_images = Cifar.quantize(self.val_images)
        # self.quantized_train_images = Cifar.quantize(self.train_images)
        # self.quantized_test_images = Cifar.quantize(self.test_images)
        #print(self.quantized_val_images[0])

        self.val_images = Cifar.normalize(self.val_images)
        self.train_images = Cifar.normalize(self.train_images)
        self.test_images = Cifar.normalize(self.test_images)
        #print(self.val_images[0]) 

    def data(self, batch_size, is_supervised, percentage=None):
        if is_supervised:
            if not percentage is None:
                length = len(self.train_images) * (percentage/100)
                length = int(length)
            else:
                length = len(self.train_images)
            indices = np.random.randint(0, length, size=batch_size)
            x = self.train_images[indices]
            y = self.train_labels[indices]
            y = Cifar.one_hot(y, 10)
            return x, y

        else:
            indices = np.random.randint(0, len(self.train_images), size=batch_size)
            x = self.train_images[indices]
            #plt.imshow(np.squeeze(x[0]))
            #plt.show()
            return x

    def val_data(self, batch_size, is_supervised):
        if is_supervised:
            indices = np.random.randint(0, len(self.val_images), size=batch_size)
            x = self.val_images[indices]
            y = self.val_labels[indices]
            y = Cifar.one_hot(y, 10)
            return x,y

        else:
            indices = np.random.randint(0, len(self.val_images), size=batch_size)
            x = self.val_images[indices]
            #plt.imshow(np.squeeze(x[0]))
            return x

    def test_data(self, test_size, is_supervised):
        start = 0
        print(len(self.test_images))
        cont = True
        while cont:
            end = start + test_size
            if end >= len(self.test_images):
                end = len(self.test_images) - 1
                cont = False
            print("start: {0}, end: {1}".format(start, end))
            x = self.test_images[start:end]
            y = self.test_labels[start:end]
            y = Cifar.one_hot(y, 10)

            if is_supervised:
                yield x, y
            else:
                yield x

            start = end