
## Widrow-Huff learning and adaptive filters
----

[image01]: ./screenshots/screenshot_5.PNG "Numpy"
[image02]: ./screenshots/screenshot_6.PNG "Tensorflow"

Assignment 03 (Due date: Oct. 8, 2017)
Neural Networks
Assignment 03
Due: Oct. 8, 2017
 
### Problem Statement
---
- The purpose of the this assignment is to practice with Widrow-Huff learning and adaptive filters.
- Write a program to predict the price and the volume of a stock by using the Adaptive filter ADALINE.

 
  **Sliders:**
       
       1] Number of Delayed Elements: This slider selects number of delayed elements for each input (price, volume). Range: 0 to 100 default: 10.
       2] Learning Rate: Adjust the learning rate. Range should be between 0.001 and 1.0. Default value = 0.1
       3] Training Sample Size (Percentage): This slider allows the user to select the percentage of the samples to be used for training. range 0% to 100%. Default value should be 80% which means 80% of the samples should be used for training and the other 20% should be used for error calculation.
       4] Batch Size:  This slider selects the batch size. Range  1 to 200. Default 100.
       5] Number of Iterations: This slider allows the user to change the number of times that the system goes over all the training samples. Range 1 to 100. Default: 10


  **Buttons:**

        1] Set Weights to Zero: When this button is pushed all the weights and biases should be set to zero.
        2] Adjust Weights: 
           - When this button is pressed the LMS algorithm should be applied and plots should be displayed. 
           - Your program should display four plots, Mean Squared Error (MSE) and Maximum Absolute Error (MAE) for price and volume. 
           - Calculations of the error should be done after the "Batch Size" samples have been processed. 
           - In other words, go through the current training batch and adjust the weight accordingly. 
           - Once that batch is processed, freeze the weights and biases, run the test set through the network and display MSE and MAE for price and volume. 
           - Notice that you will end up with four plots. The limits of the error axes should be set between 0 and 2.  

   **Notes:**
   - When your program is started it should automatically read the input data file. Normalize the price and volume data by dividing each of the values by the maximum value of the corresponding data and then subtracting 0.5 from each value.
   - Your neural network should have two nodes (Price, Volume).
   - The number of inputs to each of these nodes depends on the "Number of Delayed Elements"
   - The weights should not be reset when the "Learn" button is pressed.
   - Make sure that you follow the submission guidelines to submit your code to Blackboard
   - You should include the data file as part of your submission.
   - You may use TensorFlow for this assignment if you wish.
 
**Clarification:**
   - If the "Number of Delayed Elements" is equal to 7 then your network will have 16 input values and two outputs (one current price and 7 previous prices plus one current volume and 7 previous volumes).
   - Assuming that the total number of samples in the input file is 2000 and Training Sample Size (Percentage)  = 50% , then the  number of training samples will be equal to 1000. Assuming "Batch Size"=200, and the  "Number of Iterations"=6 ; Then you should select the first 1000 samples as training and use the rest of the samples as test. 
   - After processing every 200 training samples, you should calculate the mean and max error for price and volume (using the test samples). You should go over the 1000 training samples 6 times. 
   - In other words it will take 5 batches to go over the entire training samples, and you should go over the entire training samples 6 times (30 batches should be processed).

 
### Results 
---
- I have created a gui using tkinter to demo Widrow-Huff learning and adaptive filters error graph
- I have used tensorflow and normal numpy
- To run the program you clone the repository
- Download MNIST data and put the data in **`DATA`** folder
- run **`Shah_03_01.py`** file which will create the gui
- Below is the sceenshot of the gui

### Screenshot Error Graphs with Normal Numpy
---
![SCREEENSHOT][image01]

### Screenshot Error Graphs with Tensorflow
---
![SCREEENSHOT][image02]

