import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

def residual(net, num_filt, kernel_size, keepProb, isTraining, isFirst, isLast):
	with slim.arg_scope([slim.layers.convolution], 
		padding='SAME',
		weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
		activation_fn=None
		):

		if isFirst:
			net = tf.nn.relu(net)
		save = net
		net = slim.layers.batch_norm(net, is_training=isTraining)
		net = tf.nn.relu(net)
		net = slim.layers.convolution(net, num_filt, kernel_size, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
			activation_fn=None)
		net = slim.layers.dropout(net, keep_prob=keepProb, is_training=isTraining)
		net = slim.layers.batch_norm(net, is_training=isTraining)
		net = tf.nn.relu(net)
		net = slim.layers.convolution(net, num_filt, kernel_size, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
			activation_fn=None)
		net = save + net
		if isLast:
			net = tf.nn.relu(net)
		return net

def input_distortion(images, isTraining, batch_size):
    images = tf.cond(isTraining, lambda: tf.random_crop(images, [batch_size, 24, 24, 3]), 
        lambda: tf.map_fn(lambda image: tf.image.resize_image_with_crop_or_pad(image, target_height=24, target_width=24), images))
    images = tf.cond(isTraining, lambda: tf.map_fn(tf.image.random_flip_left_right, images), lambda: images)
    return images

class Model():

	def __init__(self, sess, data, nEpochs, learning_rate_1, learning_rate_2, batch_size, is_supervised):
		self.sess = sess
		self.data = data #initialize this with Cifar.data
		self.nEpochs = nEpochs
		self.learning_rate_1 = learning_rate_1
        self.learning_rate_2 = learning_rate_2
		self.batch_size = batch_size
		self.is_supervised = is_supervised
		self.build_model(self.is_supervised)

	def build_model(self, is_supervised):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.isTraining = tf.placeholder(tf.bool, shape=[])
        
        if is_supervised:
            self.y = tf.placeholder(tf.float32, shape=[None, 10])

        if not is_supervised:
            L, ab, L_hat, ab_hat, self.images = (slim.layers.flatten(x) for x in self.unsupervised_arch(self.x))
            self.ab_hat_l2_loss = tf.reduce_mean(tf.pow(tf.abs(ab - ab_hat), 2))
    		self.L_hat_l2_loss = tf.reduce_mean(tf.pow(tf.abs(L - L_hat), 2))

    def unsupervised_arch(images):
        images = input_distortion(images, self.isTraining, self.batch_size)
        L = images[:, : :, 0]
        ab = images[:, :, :, [1, 2]]

        with slim.arg_scope([slim.layers.convolution], 
            padding='SAME',
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            normalizer_fn = slim.layers.batch_norm,
            normalizer_params = {'is_training', self.isTraining}
            ):

            ab_hat = slim.layers.convolution(L, 32, [3, 3], scope='L_conv1') # 32 x 32 x 32
            ab_hat = slim.layers.max_pool2d(ab_hat, [2, 2]) # 16 x 16 x 32
            ab_hat = slim.layers.convolution(ab_hat, 64, [3, 3], scope='L_conv2') # 16 x 16 x 64
            with tf.variable_scope('L_res1'):
                ab_hat = residual(ab_hat, 64, [3, 3], 0.7, self.isTraining, True, False) # 16 x 16 x 64

            with tf.variable_scope('L_res2'):
                ab_hat = residual(ab_hat, 64, [3, 3], 0.7, self.isTraining, False, True) # 16 x 16 x 64

            ab_hat = slim.layers.convolution(ab_hat, 2, [1, 1], scope='L_conv3', activation_fn=None) # 16 x 16 x 2

        with slim.arg_scope([slim.layers.convolution], 
            padding='SAME',
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            normalizer_fn = slim.layers.batch_norm,
            normalizer_params = {'is_training', self.isTraining}
            ):

            L_hat = slim.layers.convolution(ab, 32, [3, 3], scope='ab_conv1') # 32 x 32 x 32
            L_hat = slim.layers.max_pool2d(L_hat, [2, 2]) # 16 x 16 x 32
            L_hat = slim.layers.convolution(L_hat, 64, [3, 3], scope='ab_conv2') # 16 x 16 x 64
            with tf.variable_scope('ab_res1'):
                L_hat = residual(L_hat, 64, [3, 3], 0.7, self.isTraining, True, False) # 16 x 16 x 64

            with tf.variable_scope('ab_res2'):
                L_hat = residual(L_hat, 64, [3, 3], 0.7, self.isTraining, False, True) # 16 x 16 x 64

            L_hat = slim.layers.convolution(L_hat, 1, [1, 1], scope='L_conv3', activation_fn=None) # 16 x 16 x 1

        L = tf.image.resize_bilinear(L, [16, 16])
        ab = tf.image.resize_bilinear(ab, [16, 16])

        return L, ab, L_hat, ab_hat, images

    def train_init(self):
        model_variables = slim.get_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if update_ops:
            updates = tf.group(*update_ops)

            self.optim = tf.group(updates,
                tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate_1)
                    .minimize(self.ab_hat_l2_loss)
                ,
                tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate_2)
                    .minimize(self.L_hat_l2_loss)
                )
        else:
            self.optim = tf.group(
                tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate_1)
                    .minimize(self.ab_hat_l2_loss)
                ,
                tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate_2)
                    .minimize(self.L_hat_l2_loss)
                )

        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, x, y=None):
        if self.is_supervised:
            if not y:
                raise ValueError("Must supply labels for supervised training")

            # NEED TO FILL IN
        else:
            if y:
                raise ValueError("Do not supply labels for unsupervised training")

            ab_hat_l2_loss, L_hat_l2_loss, _, ims = sess.run(
                [self.ab_hat_l2_loss, self.L2_hat_l2_loss, self.optim, self.images], 
                feed_dict = {self.x: x, self.isTraining: True}
                )
            print('ab_hat_l2loss: {0}, L_hat_l2_loss: {1}'.format(ab_hat_l2_loss, L_hat_l2_loss))
            plt.imshow(np.squeeze(ims[0]))
            plt.show()

    def train(self):
        for epoch in range(self.nEpochs):
            if self.is_supervised:
                for x, y in self.data(self.batch_size, self.is_supervised):
                    self.train_iter(x, y)
            else:
                for x in self.data(self.batch_size, self.is_supervised):
                    self.train_iter(x)


