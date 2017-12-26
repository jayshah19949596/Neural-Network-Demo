
## Hebbian learning rules (Due date: Sept. 27, 2017)
----

[image01]: ./screenshot/screenshot.JPG "screenshot"


Neural Networks
Assignment 02
Due: Sept. 27, 2017
 
### Problem Statement
---
- The purpose of the this assignment is to practice with Hebbian learning rules.
- Write a program to implement a single layer neural network with 10 nodes.
- Your program should include 1 sliders, 2 buttons, and 2 drop-down selection box.
- **Slider:**

      1. "alpha (learning rate)". Range should be between 0.001 and 1.0. Default value = 0.1
      
 - **Buttons:**
 
       1. "Adjust Weights (Learn)". When this button is pressed the selected Hebbian learning rule should  be applied for 100 epochs.
       2. "Rondomize Weights". When this button is pressed weights and biases should be randomized. The value of the randomized weights and biases should be from -0.001 to 0.001 .
 
- **Drop-Down Selection Boxes**

      1. "Select Learning Method". A drop-down box to select one of the three Hebbian rules from your textbook, i.e. "Filtered Learning", "Delta Rule", or "Unsupervised Hebb".
      2. "Transfer Functions". A drop-down box to allow the user to select between three transfer functions (Symmetrical Hard limit, Hyperbolic Tangent, and Linear)

- **Notes:**

  - Put all the mnist image files in a directory called "Data". This directory should be located in the same folder as your main program. When submitting your assignment you do not need to submit the mnist images.
  - When your program starts it should automatically read the mnist data. Once the mnist data is read your program should convert each image to a vector and normalize each vector element to be in the range of -1 to 1. i.e. , divide the input numbers by 127.5 and subtract 1.0. Notice that normalizing each element of a vector to be between -1 to 1 does not normalize the vector itself.
  - When your program starts it should automatically initialize all the weights and biases to be from -0.001 to 0.001
Separate the input data into two sets. The first set should include 80% of your data set (randomly selected). This is your training set. 
  - The second set (the other 20%) is the test set. The test set will be used for calculating the error rate.
  - The index of the maximum value of the output vector is selected as the class id.
  - Plot the error rate, in percent, after each epoch on the error rate graph. An epoch is one pass over all the training samples. 
  - This means that you train (adjust the weights) for one complete set of the training data (one epoch). Then turn off the training (freeze the weights and biases) and run the test data through the network and calculate the error rate
  - The error-rate graph should be able to display up to 1000 epochs.
  - If you are using python, your assignment should run on Python 3.x
  - Make sure that you follow the submission guidelines to submit your code to Blackboard.
 
 
### Results 
---

- I have created a gui using tkinter to demo hebbian learning rule error graph
- To run the program you clone the repository
- run **`Shah_02_01.py`** file which will create the gui
- Below is the sceenshot of the gui


![SCREEENSHOT][image01]
