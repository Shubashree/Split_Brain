import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cifar import Cifar
from model import Model

cifar_data = Cifar('../hw3/cifar10')

sess = tf.Session()
model = Model(
	sess=sess,
	data=cifar_data.data,
	val_data = cifar_data.val_data,
	nEpochs=10,
	learning_rate_1=1e-2,
	learning_rate_2=1e-2,
	batch_size=64,
	is_supervised=False
	)

model.train_init()
model.train()