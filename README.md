# Split Brain Autoencoder

Implementation of [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction] (https://arxiv.org/abs/1611.09842) by Zhang et al

## cifar.py 
* Imports CIFAR-10 data
* Converts from RGB colorspace to LAB colorspace
* Normalizes each channel to [0,1]
* Quantizes, or Bins, each channel in order to train under classification loss. "L" quantizes to 100 bins, and "ab" quantizes to a 16x16 grid
  
## model.py

* Builds unsupervised model. Consists of two convolutional neural networks, one predicts the "ab" channel from the "L" channel, other predicts the "L" channel from the "ab" channel. Trained using the Adam Optimizer under classification loss, where predictions were compared to the ground truth. Tuned until plausible colorizations were produced. 
* Saves unsupervised model as a checkpoint. Restores model if selected
* Builds supervised model. If pre-trained option is selected, the trained unsupervised model parameters will be restored and a linear classifier will be trained on 10% of the training data at the feature extraction layer. If untrained option is selected, the entire model will be trained on 10% of the training data. 

## midterm.py
* Serves as the test bench


  
  


