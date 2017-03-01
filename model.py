import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


###########################################################
# L : [0, 100], A: [-86.185, 98.254], B: [-107.863, 94.482]
###########################################################

###########################################################
# CLEAR TRAINING LOG TENSORBOARD DIRECTORY BEFORE RUNNING #
###########################################################


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

    def __init__(self, sess, data, val_data, num_iter, sup_learning_rate, uns_learning_rate_1, uns_learning_rate_2, batch_size, 
        is_supervised, is_untrained):
        self.sess = sess
        self.data = data #initialize this with Cifar.data
        self.val_data = val_data
        self.num_iter = num_iter
        self.sup_learning_rate = sup_learning_rate
        self.uns_learning_rate_1 = uns_learning_rate_1
        self.uns_learning_rate_2 = uns_learning_rate_2
        self.batch_size = batch_size
        self.is_supervised = is_supervised
        self.build_model(self.is_supervised)
        self.sup_percentage = None
        self.is_untrained = is_untrained

    def change_sup_percentage(self, percentage):
        self.sup_percentage = percentage

    def build_model(self, is_supervised):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.isTraining = tf.placeholder(tf.bool, shape=[])
        
        if is_supervised:
            self.y = tf.placeholder(tf.float32, shape=[None, 10])
            result = self.unsupervised_arch(self.x)

            self.L_feature_map = result[2]
            self.ab_feature_map = result[3]

            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, "./model.ckpt")

            self.prediction = self.supervised_arch(self.L_feature_map, self.ab_feature_map)
            tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.prediction)
            self.sup_loss = tf.losses.get_total_loss()
            correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=1), tf.argmax(input=self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            if is_untrained:
                train_loss_sum = tf.summary.scalar('Supervised Untrained Training Loss', self.sup_loss)
                train_acc_sum = tf.summary.scalar('Supervised Untrained Training Accuracy', self.accuracy)

                val_loss_sum = tf.summary.scalar('Supervised Untrained Val Loss', self.sup_loss)
                val_acc_sum = tf.summary.scalar('Supervised Untrained Val Accuracy', self.accuracy)

                self.train_merged = tf.summary.merge([train_loss_sum, train_acc_sum])
                self.val_merged = tf.summary.merge([val_loss_sum, val_acc_sum])
                self.log_writer = tf.summary.FileWriter('./train_unt_sup_logs', self.sess.graph)                

            else:     
                train_loss_sum = tf.summary.scalar('Supervised Trained Training Loss', self.sup_loss)
                train_acc_sum = tf.summary.scalar('Supervised Trained Training Accuracy', self.accuracy)

                val_loss_sum = tf.summary.scalar('Supervised Trained Val Loss', self.sup_loss)
                val_acc_sum = tf.summary.scalar('Supervised Trained Val Accuracy', self.accuracy)

                self.train_merged = tf.summary.merge([train_loss_sum, train_acc_sum])
                self.val_merged = tf.summary.merge([val_loss_sum, val_acc_sum])
                self.log_writer = tf.summary.FileWriter('./train_tra_sup_logs', self.sess.graph)

        if not is_supervised:
            result = self.unsupervised_arch(self.x)
            L, ab, L_hat, ab_hat = (slim.layers.flatten(x) for x in result[0])
            #L, ab, L_hat, ab_hat = result[0]
            self.images = result[1]
            self.ab_hat_l2_loss = tf.reduce_mean(tf.pow(tf.abs(ab - ab_hat), 2))
            self.L_hat_l2_loss = tf.reduce_mean(tf.pow(tf.abs(L - L_hat), 2))

            #TensorBoard Logging:
            train_ab_sum = tf.summary.scalar('Unsupervised Training ab_hat L2 loss', self.ab_hat_l2_loss)
            train_l_sum = tf.summary.scalar('Unsupervised Training L hat L2 loss', self.L_hat_l2_loss)

            val_ab_sum = tf.summary.scalar('Unsupervised Val ab_hat L2 loss', self.ab_hat_l2_loss)
            val_l_sum = tf.summary.scalar('Unsupervised Val L hat L2 loss', self.L_hat_l2_loss)

            self.train_merged = tf.summary.merge([train_ab_sum, train_l_sum])
            self.val_merged = tf.summary.merge([val_ab_sum, val_l_sum])
            self.log_writer = tf.summary.FileWriter('./train_uns_logs', self.sess.graph)

            self.saver = tf.train.Saver()
    
    #TODO: MAKE SURE ABOUT VARIABLE COLLECTIONS WHEN PASSING TO TRAINING OP
    #TODO: Check update collections, how to only pass a specific collection
    #TODO: Figure out how to fetch specific variable names
    def supervised_arch(self, L_feature_map, ab_feature_map):
        self.total_features = tf.concat(
            [L_feature_map, ab_feature_map],
            axis=3
            ) # 12 x 12 x 128

        with tf.variable_scope('Supervised'):
            with slim. arg_scope([slim.layers.convolution, slim.layers.fully_connected],
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                normalizer_fn = slim.layers.batch_norm,
                normalizer_params = {'is_training': self.isTraining, 'updates_collections': ['supervised_update_coll']},
                variables_collections = ['supervised_var_coll']
                ):
                result = slim.layers.convolution(self.total_features, 64, [3, 3], scope='S_conv1') # 12 x 12 x 64
                result = slim.layers.flatten(result)
                result = slim.layers.fully_connected(result, 1024, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
                result = slim.layers.dropout(result, keep_prob=0.5, is_training=self.isTraining)
                result = slim.layers.fully_connected(result, 10, weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
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
        if self.is_supervised:
            if self.is_untrained:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    self.optim = tf.group(updates,
                        tf.train.AdamOptimizer(
                            learning_rate=self.sup_learning_rate
                            )
                            .minimize(self.sup_loss)
                        )
                else:
                    self.optim = tf.train.AdamOptimizer(
                        learning_rate=self.sup_learning_rate,
                        ).minimize(self.sup_loss)
            else:
                update_ops = tf.get_collection('supervised_update_coll')
                model_variables = tf.get_collection('supervised_var_coll')
                if update_ops:
                    updates = tf.group(*update_ops)
                    self.optim = tf.group(updates,
                        tf.train.AdamOptimizer(
                            learning_rate=self.sup_learning_rate
                            )
                            .minimize(self.sup_loss, var_list=model_variables)
                        )
                else:
                    self.optim = tf.train.AdamOptimizer(
                        learning_rate=self.sup_learning_rate,
                        ).minimize(self.sup_loss, var_list=model_variables)

        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)

                self.optim = tf.group(updates,
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_1)
                        .minimize(self.ab_hat_l2_loss)
                    ,
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_2)
                        .minimize(self.L_hat_l2_loss)
                    )
            else:
                self.optim = tf.group(
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_1)
                        .minimize(self.ab_hat_l2_loss)
                    ,
                    tf.train.AdamOptimizer(
                        learning_rate=self.uns_learning_rate_2)
                        .minimize(self.L_hat_l2_loss)
                    )

        self.sess.run(tf.global_variables_initializer())

    def train_iter(self, iteration, x, y=None):
        if self.is_supervised:
            if y is None:
                raise ValueError("Must supply labels for supervised training")
            loss, _, accuracy, summary = self.sess.run(
                [self.sup_loss, self.optim, self.accuracy, self.train_merged],
                feed_dict = {self.x: x, self.y: y, self.isTraining: True}
                )

            print('SUP--  loss: {0}, accuracy: {1}'.format(loss, accuracy))
            self.log_writer.add_summary(summary, iteration)

            # NEED TO FILL IN
        else:
            if y:
                raise ValueError("Do not supply labels for unsupervised training")

            ab_hat_l2_loss, L_hat_l2_loss, _, summary, ims = self.sess.run(
                [self.ab_hat_l2_loss, self.L_hat_l2_loss, self.optim, self.train_merged, self.images], 
                feed_dict = {self.x: x, self.isTraining: True}
                )
            print('ab_hat_l2loss: {0}, L_hat_l2_loss: {1}, ITERATION: {2}'.format(ab_hat_l2_loss, L_hat_l2_loss, iteration))
            self.log_writer.add_summary(summary, iteration)
            # plt.imshow(np.squeeze(ims[0]))
            # plt.show()

    def info_iter(self, iteration, x, y=None):
        if self.is_supervised:
            if y is None:
                raise ValueError("Must supply labels for supervised training")

            loss, accuracy, summary = self.sess.run(
                [self.sup_loss, self.accuracy, self.val_merged],
                feed_dict = {self.x: x, self.y: y, self.isTraining:False}
                )

            print('SUP-- VAL: loss:{0}, accuracy: {1}'.format(loss, accuracy))
            self.log_writer.add_summary(summary, iteration)

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

    def train(self):
        for iteration in range(self.num_iter):
            if self.is_supervised:
                x, y= self.data(self.batch_size, self.is_supervised, self.sup_percentage)
                self.train_iter(iteration, x, y)

                if iteration % 100 == 0:
                    self.info_iter(iteration, x, y)

            else:
                x = self.data(self.batch_size, self.is_supervised)
                self.train_iter(iteration, x)

                if iteration % 100 == 0:
                    self.info_iter(iteration, x)

        if not self.is_supervised:
            save_path = self.saver.save(self.sess, "./model.ckpt")
            print("Model saved in file: %s" % save_path)

                    