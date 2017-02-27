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

    def __init__(self, sess, data, val_data, nEpochs, learning_rate_1, learning_rate_2, batch_size, is_supervised, saver):
        self.sess = sess
        self.data = data #initialize this with Cifar.data
        self.val_data = val_data
        self.nEpochs = nEpochs
        self.learning_rate_1 = learning_rate_1
        self.learning_rate_2 = learning_rate_2
        self.batch_size = batch_size
        self.is_supervised = is_supervised
        self.build_model(self.is_supervised)
        self.saver = saver

    def build_model(self, is_supervised):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.isTraining = tf.placeholder(tf.bool, shape=[])
        
        if is_supervised:
            self.y = tf.placeholder(tf.float32, shape=[None, 10])
            result = self.unsupervised_arch(self.x)

            self.L_feature_map = result[2]
            self.ab_feature_map = result[3]

            self.saver.restore(self.sess, "./model.ckpt")

            result = supervised_arch(self.L_feature_map, self.ab_feature_map)


        if not is_supervised:
            result = self.unsupervised_arch(self.x)
            L, ab, L_hat, ab_hat = (slim.layers.flatten(x) for x in result[0])
            #L, ab, L_hat, ab_hat = result[0]
            self.images = result[1]
            self.ab_hat_l2_loss = tf.reduce_mean(tf.pow(tf.abs(ab - ab_hat), 2))
            self.L_hat_l2_loss = tf.reduce_mean(tf.pow(tf.abs(L - L_hat), 2))

            #TensorBoard Logging:
            train_ab_sum = tf.summary.scalar('Training ab_hat L2 loss', self.ab_hat_l2_loss)
            train_l_sum = tf.summary.scalar('Training L hat L2 loss', self.L_hat_l2_loss)

            val_ab_sum = tf.summary.scalar('Val ab_hat L2 loss', self.ab_hat_l2_loss)
            val_l_sum = tf.summary.scalar('Val L hat L2 loss', self.L_hat_l2_loss)

            self.train_merged = tf.summary.merge([train_ab_sum, train_l_sum])
            self.val_merged = tf.summary.merge([val_ab_sum, val_l_sum])
            self.log_writer = tf.summary.FileWriter('./train_logs', self.sess.graph)

    
    #TODO: MAKE SURE ABOUT VARIABLE COLLECTIONS WHEN PASSING TO TRAINING OP
    #TODO: Check update collections, how to only pass a specific collection
    #TODO: Figure out how to fetch specific variable names
    def supervised_arch(self, L_feature_map, ab_feature_map):
        self.total_features = tf.concat(
            [Lfeature_map, ab_feature_map],
            axis=3
            ) # 12 x 12 x 128

        with tf.variable_scope('Supervised'):
            result = slim.layers.convolution(self.total_features, 64, [3, 3], scope='S_conv1') # 12 x 12 x 64
            result = slim.layers.flatten(result)
            result = slim.layers.fully_connected(net, 1024, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
            result = slim.layers.dropout(net, keep_prob=0.5, is_training=self.isTraining)
            result = slim.layers.fully_connected(net, 10, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                activation_fn=None)

        return result

    def unsupervised_arch(self, images):
        images = input_distortion(images, self.isTraining, self.batch_size)
        L = tf.reshape(images[:, :, :, 0], shape=[-1, 24, 24, 1])
        ab = tf.concat(
            [tf.reshape(images[:, :, :, 1], shape=[-1, 24, 24, 1]), tf.reshape(images[:, :, :, 2], shape=[-1, 24, 24, 1])],
            axis=3
            )

        with slim.arg_scope([slim.layers.convolution], 
            padding='SAME',
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            normalizer_fn = slim.layers.batch_norm,
            normalizer_params = {'is_training': self.isTraining}
            ):

            ab_hat = slim.layers.convolution(L, 32, [3, 3], scope='L_conv1') # 24 x 24 x 32
            ab_hat = slim.layers.max_pool2d(ab_hat, [2, 2]) # 12 x 12 x 32
            ab_hat = slim.layers.convolution(ab_hat, 64, [3, 3], scope='L_conv2') # 12 x 12 x 64
            with tf.variable_scope('L_res1'):
                ab_hat = residual(ab_hat, 64, [3, 3], 0.7, self.isTraining, True, False) # 12 x 12 x 64

            with tf.variable_scope('L_res2'):
                ab_hat = residual(ab_hat, 64, [3, 3], 0.7, self.isTraining, False, True) # 12 x 12 x 64

            ### PUT THIS LINE WHERE YOU WANT TO EXTRACT SUPERVISED AB FEATURES ###
            ab_features = ab_hat

            ab_hat = slim.layers.convolution(ab_hat, 2, [1, 1], scope='L_conv3', activation_fn=None) # 12 x 12 x 2

        with slim.arg_scope([slim.layers.convolution], 
            padding='SAME',
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            normalizer_fn = slim.layers.batch_norm,
            normalizer_params = {'is_training': self.isTraining}
            ):

            L_hat = slim.layers.convolution(ab, 32, [3, 3], scope='ab_conv1') # 24 x 24 x 32
            L_hat = slim.layers.max_pool2d(L_hat, [2, 2]) # 12 x 12 x 32
            L_hat = slim.layers.convolution(L_hat, 64, [3, 3], scope='ab_conv2') # 12 x 12 x 64
            with tf.variable_scope('ab_res1'):
                L_hat = residual(L_hat, 64, [3, 3], 0.7, self.isTraining, True, False) # 12 x 12 x 64

            with tf.variable_scope('ab_res2'):
                L_hat = residual(L_hat, 64, [3, 3], 0.7, self.isTraining, False, True) # 12 x 12 x 64

            ### PUT THIS LINE WHERE YOU WANT TO EXTRACT SUPERVISED L FEATURES ###
            L_features = L_hat

            L_hat = slim.layers.convolution(L_hat, 1, [1, 1], scope='ab_conv3', activation_fn=None) # 12 x 12 x 1

        L = tf.image.resize_bilinear(L, [12, 12])
        ab = tf.image.resize_bilinear(ab, [12, 12])

        return [(L, ab, L_hat, ab_hat), images, L_features, ab_features]

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

    def train_iter(self, iteration, x, y=None):
        if self.is_supervised:
            if not y:
                raise ValueError("Must supply labels for supervised training")

            # NEED TO FILL IN
        else:
            if y:
                raise ValueError("Do not supply labels for unsupervised training")

            ab_hat_l2_loss, L_hat_l2_loss, _, summary, ims = self.sess.run(
                [self.ab_hat_l2_loss, self.L_hat_l2_loss, self.optim, self.train_merged, self.images], 
                feed_dict = {self.x: x, self.isTraining: True}
                )
            print('ab_hat_l2loss: {0}, L_hat_l2_loss: {1}'.format(ab_hat_l2_loss, L_hat_l2_loss))
            self.log_writer.add_summary(summary, iteration)
            # plt.imshow(np.squeeze(ims[0]))
            # plt.show()

    def info_iter(self, iteration, x, y=None):
        if self.is_supervised:
            if not y:
                raise ValueError("Must supply labels for supervised training")

            # NEED TO FILL IN
        else:
            if y:
                raise ValueError("Do not supply labels for unsupervised training")

            ab_hat_l2_loss, L_hat_l2_loss, summary = self.sess.run(
                [self.ab_hat_l2_loss, self.L_hat_l2_loss, self.val_merged],
                feed_dict = {self.x: x, self.isTraining: False}
                )
            print('VAL: ab_hat_l2_loss: {0}, L_hat_l2_loss: {1}'.format(ab_hat_l2_loss, L_hat_l2_loss))
            self.log_writer.add_summary(summary, iteration)

    def unsupervised_train(self):
        iteration = 0
        for epoch in range(self.nEpochs):
            if self.is_supervised:
                for x, y in self.data(self.batch_size, self.is_supervised):
                    self.train_iter(iteration, x, y)

                    if iteration % 100 == 0:
                        self.info_iter(iteration, x, y)

                    iteration += 1
            else:
                for x in self.data(self.batch_size, self.is_supervised):
                    self.train_iter(iteration, x)

                    if iteration % 100 == 0:
                        self.info_iter(iteration, x)
                    iteration += 1

        save_path = self.saver.save(self.sess, "./model.ckpt")
        print("Model saved in file: %s" % save_path)
                    