import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cifar import Cifar
from model import Model

cifar_data = Cifar('../hw3/cifar10')
cifar_data.convert_to_lab()

sess = tf.Session()
model = Model(
	sess=sess,
	data=cifar_data.data,
	val_data = cifar_data.val_data,
	test_data = cifar_data.test_data,
	num_iter=3000,
	sup_learning_rate=1e-2,
	uns_learning_rate_1=1e-2,
	uns_learning_rate_2=1e-2,
	batch_size=32,
	test_size=100,
	is_supervised=False,
	is_untrained=False
	)

#model.change_sup_percentage(100)
model.train_init()
model.train()
# model.saver.restore(model.sess, './saved_uns_model/model.ckpt')
model.test()