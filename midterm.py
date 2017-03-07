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
	num_iter=1000,
	sup_learning_rate=1e-2,
	uns_learning_rate_1=None,
	uns_learning_rate_2=None,
	batch_size=64,
	test_size=100,
	is_supervised=True,
	is_untrained=False
	)

model.change_sup_percentage(100)
model.train_init()
model.train()
model.test()