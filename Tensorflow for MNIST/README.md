
## Back-Propagation neural networks with TENSORFLOW 
----

[image01]: ./screenshots/MSE.JPG "MSE"
[image02]: ./screenshots/SoftMax.JPG "SoftMax"

Assignment 04 (Due date: Oct. 25, 2017)
Neural Networks
Assignment 04
Due date: Oct. 25, 2017
 
### Problem Statement
---
 - The purpose of the this assignment is to practice with back-propagation neural networks.
 
 - Implement a fully-connected back-propagation network using TensorFlow:
 - Your network should include one hidden layer and one output layer with 10 nodes. The input for the neural network is the mnist data. 
 - Note that the mnist data includes about 60,000 training and about 10,000 test images.
 - Your program should include the following sliders, buttons, and drop-down box.
 
 **Sliders:**
 
   1] "Alpha": (Learning rate) Range should be between 0.001 and 1.0. Default value = 0.1 increments=.001
   2] "Lambda": (Weight regularization). Range should be between 0.0 and 1.0. Default value = 0.01 Increments=0.01
   3] "Num. of Nodes"": (number of nodes in the hidden layer). Range 1 to 500. Default value=100  increment=1
   4] "Batch Size": Number of samples to process in each batch. Range 1 to 256. Default value=64 Increments=1
   5] "Percent Data": Percent of training and test data points to be used. Range 1 to 100. Default value=10, increments=1

  **Buttons:**
  
   1] "Adjust Weights": (Learn) When this button is pressed the gradient descent should be applied 10 epochs. The graphs should not be cleared when this button is pressed.
   2] "Reset Weights". When this button is pressed all weights should be set to zeros. and all graphs should be cleared.
 
  **Drop-Down Selection Box**
  
   1] "Hidden Layer Transfer Function". A drop-down box to allow the user to select between two transfer functions for the hidden layer (Relu, and Sigmoid)
   2] "Output Layer Transfer Function". A drop-down box to allow the user to select between two transfer functions for the output layer (Softmax or Sigmoid)
   3] "Cost Function". A drop-down box to allow the user to select between two cost functions (Cross Entropy or MSE)

  **Notes:**
  
  - The mnist data includes two sets. "Train" set and "Test" set. Put all the mnist image files (which includes two "Train" and   - "Test" directory in a directory called "MNIST_data". This directory should be located in the same folder as your main program.
  - When submitting your assignment DO NOT submit the "MNIST_data" directory. Use a relative path to "MNIST_data" directory.
  - When your program starts it should automatically read the "Percent Data" from "Train" and "Test" directories. 
  - Once the mnist data is read, your program should convert each image to a vector and normalize each vector element to be in the range of -1 to 1. i.e. , divide the input numbers by 127.5 and subtract 1. 
  - Notice that normalizing each element of a vector to be between -1 to 1 does not normalize the vector itself.
  - When your program starts it should automatically initialize all the weights and biases to values between -0.0001 and 0.0001.
  - The target vector is a 10 by 1 vector with the value of the class index set to 1 and all other values set to 0. For example if the class of the input image is "3" then the target vector should be [0,0,0,1,0,0,0,0,0,0].T
  - Plot the error rate, in percent, after each epoch on the error rate graph. An epoch is one pass over all the training samples. 
  - This means that you train (adjust the weights) for one complete set of the training data (one epoch). Then turn off the training (freeze the weights and biases) and run the test data through the network and calculate the error rate.
  - Plot the loss function after each epoch on a separate graph.
  - The error-rate graph should be able to display up to 100 epochs.
  - Show the confusion matrix as an image. You may use the code in the following link: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
  - If you are using python, your assignment should run on Python 3.x
  - Make sure that you follow the submission guidelines to submit your code to Blackboard.
 
### Results 
---
- I have created a gui using tkinter to demo MSE error graph and Softmax error graph
- I have used tensorflow for mnist classification
- To run the program you clone the repository
- run **`Shah_04_01.py`** file which will create the gui
- Below is the sceenshot of the gui

### Screenshot for MEAN SQUARE ERROR 
---
![SCREEENSHOT][image01]

### Screenshot for SOFTMAX
---
![SCREEENSHOT][image02]

