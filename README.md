# Split Brain Autoencoder

Implementation of [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction by Zhang et al] (https://arxiv.org/abs/1611.09842)

Training a Split-Brain Autoencoder on CIFAR-10 images.

Unsupervised training: loss function compares prediction with a 12x12 downsampled version of the original image.
Supervised training: keeping a subset of the layers fully trained via unsupervised learning, and then training additional layers via supervised learning (using a subset of the data).
