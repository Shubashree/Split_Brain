# Split Brain Autoencoder

Implementation of [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction] (https://arxiv.org/abs/1611.09842) by Zhang et al

## cifar.py 
* Imports CIFAR-10 data
* Converts from RGB colorspace to LAB colorspace
* Normalizes each channel to [0,1]
* Quantizes, or Bins, each channel in order to train under classification loss. "L" quantizes to 100 bins, and "ab" quantizes to a 16x16 grid
  
## model.py


  
  


