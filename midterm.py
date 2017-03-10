import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cifar import Cifar
from model import Model

cifar_data = Cifar('../hw3/cifar10')
#cifar_data.compute_maxs(cifar_data.train_images)
cifar_data.convert_to_lab()

sess = tf.Session()
model = Model(
	sess=sess,
	data=cifar_data.data,
	val_data = cifar_data.val_data,
	test_data = cifar_data.test_data,
	cifar=cifar_data,
	num_iter=300,
	sup_learning_rate=1e-2,
	uns_learning_rate_1=1e-2,
	uns_learning_rate_2=1e-2,
	batch_size=16,
	test_size=16,
	is_supervised=False,
	is_untrained=True	
	)

#model.change_sup_percentage(10)
model.train_init()
model.train()
# model.saver.restore(model.sess, './saved_uns_model/model.ckpt')
model.test()