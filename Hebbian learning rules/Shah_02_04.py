# Jai, Shah
# 1001-380-311
# 2017-09-27
# Assignment_02_04
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from numpy import *
import math
from skimage.io import imread_collection
from PIL import Image
import glob
import cv2
import scipy.misc
from sklearn.utils import shuffle
from sklearn import preprocessing


class Model(object):
	def __init__(self, file, activation="Hyperbolic Tangent", learning_method="Delta Rule", learning_rate=00.1):
		self.train_data, self.test_data, self.train_labels, self.test_labels \
			= load_images(file)  # (800, 784) | (200, 784) | (800,) | (200,)
		self.weights = np.random.uniform(-0.1, 0.1, (np.unique(self.train_labels).size, self.train_data.shape[1]))
		self.bias = random.uniform(-0.1, 0.1, np.unique(self.train_labels).size)
		self.learning_rate = learning_rate
		self.activation = activation
		self.learning_method = learning_method
		self.lb = preprocessing.LabelBinarizer()
		self.lb.fit(self.train_labels)
		self.epochs = []
		self.error = []
		self.total_epochs = 0
		self.iteration = 0
		print(self.learning_method, self.activation, self.learning_rate)
		
	def start_learning(self):
		start = self.iteration
		self.total_epochs += 1
		end = 0 + self.total_epochs*100
		print("start", start)
		print("end", end)
		
		for i in range(start, end):
			for j in range(0, self.train_data.shape[0]):  # self.train_data.shape[0]):
				# ==========================
				# Calculating label_vector
				# ==========================
				label_vector = self.lb.transform(np.asmatrix(self.train_labels[j])).flatten()
				label_vector = label_vector.astype(float)
				# label_vector[label_vector == 0] = -1
				
				# =======================
				# Calculating net values
				# =======================
				if self.learning_method == "Delta Rule" or self.learning_method == "Unsupervised Heb":
					net_values = self.weights.dot(np.transpose(self.train_data[j, :]))  # (10 x 784) * (784 x 1)
					net_values += self.bias  # (10, )
					if self.activation == "Hyperbolic Tangent":
						maxi = np.max(net_values)
						net_values = net_values/maxi
					# ==========================
					# Calculating output_vector
					# ==========================
					activation_vector = get_activation(self.activation, net_values)  # ()
					
					if self.learning_method == "Delta Rule":
						idx = np.argmax(activation_vector)
						output_vector = np.zeros(activation_vector.shape)
						# output_vector -= 1
						output_vector[idx] = 1
					
						# ==========================
						# Calculating error_vector
						# ==========================
						error_vector = label_vector - output_vector
					elif self.learning_method == "Unsupervised Heb":
						output_vector = activation_vector
						
				# ==========================
				# Updating Weights
				# ==========================
				if self.learning_method == "Delta Rule":
					self.update_weights_by_delta_rule(self.train_data[j, :], error_vector)
				elif self.learning_method == "Filtered Learning":
					self.update_weights_by_filtered_learning(self.train_data[j, :], label_vector)
				elif self.learning_method == "Unsupervised Heb":
					self.update_weights_by_unsupervised_heb(self.train_data[j, :], output_vector)
					
			self.test(i)
		# print(self.epochs)
		# print(len(self.error))
		self.iteration = end
	# plot_error(self.epochs, self.error)
	
	def update_weights_by_delta_rule(self, input_vector, error):
		self.bias += self.learning_rate * error.reshape((10, ))
		error = np.transpose(np.asmatrix(error))
		input_vector = np.asmatrix(input_vector)
		self.weights += self.learning_rate * (np.dot(error, input_vector))
	
	def update_weights_by_filtered_learning(self, input_vector, label_vector):
		gamma = 0.1
		self.bias = (1 - gamma) * self.bias + self.learning_rate * label_vector
		self.weights = (1 - gamma) * self.weights + self.learning_rate * np.dot(label_vector[0], input_vector)
		# self.weights[0, :] = (1 - gamma) * self.weights[0, :] + self.learning_rate * (np.dot(label_vector[0], input_vector))
		
	def update_weights_by_unsupervised_heb(self, data_input, output):
		self.bias += self.learning_rate * output
		self.weights += self.learning_rate * np.transpose(np.asmatrix(output)).dot(np.asmatrix(data_input))
	
	def test(self, epoch):

		correct = 0
		for j in range(0, self.test_data.shape[0]):  # self.test_data.shape[0]):
			net_values = np.dot(self.weights, np.transpose(self.test_data[j, :]))  # (10 x 784) * (784 x 1)
			net_values += self.bias
			if self.activation == "Hyperbolic Tangent":
				maxi = np.max(net_values)
				net_values = net_values / maxi
			
			# ==========================
			# Calculating output_vector
			# ==========================
			if self.learning_method == "Delta Rule" or self.learning_method == "Unsupervised Heb"\
			and self.activation != "Symmetrical Hard limit" :
				activation_vector = get_activation(self.activation, net_values)  # ()
				idx = np.argmax(activation_vector)
			else:
				activation_vector = net_values
				idx = np.argmax(activation_vector)
			output_vector = np.zeros(activation_vector.shape)
			# output_vector -= 1
			output_vector[idx] = 1
			
			# ==========================
			# Calculating label_vector
			# ==========================
			label_vector = self.lb.transform(np.asmatrix(self.test_labels[j])).flatten()
			# label_vector[label_vector == 0] = -1

			if (label_vector.astype(int) == output_vector.astype(int)).all():
				correct += 1
				
		error = 1 - (correct / self.test_data.shape[0])
		# print("epoch =", epoch, "error =", error)
		# print("correct = ", correct)
		self.epochs.append(epoch)
		self.error.append(error)


def plot_error(epoch, error):
	figure = plt.figure("")
	axes = figure.gca()
	axes.set_xlabel('Epochs')
	axes.set_ylabel('Error')
	axes.set_title("Error Graph")
	plt.xlim(0, len(epoch))
	plt.ylim(-0.2, 1.0)
	axes.plot(epoch, error)
	plt.show()
	

def get_activation(activation_function, net_value):
	if activation_function == 'Sigmoid':
		# =============================
		# Calculate Sigmoid Activation
		# =============================
		activation = 1.0 / (1 + np.exp(-net_value))
	
	elif activation_function == "Linear":
		# =============================
		# Calculate Linear Activation
		# =============================
		activation = net_value
	
	elif activation_function == "Symmetrical Hard limit":
		# =============================================
		# Calculate Symmetrical Hard limit Activation
		# =============================================
		if net_value.size > 1:
			activation = net_value
			activation[activation >= 0] = 1.0
			activation[activation < 0] = -1.0
		# =============================================
		# If net value is single number
		# =============================================
		elif net_value.size == 1:
			if net_value < 0:
				activation = -1.0
			else:
				activation = 1.0
	
	elif activation_function == "Hyperbolic Tangent":
		# =============================================
		# Calculate Hyperbolic Tangent Activation
		# =============================================
		activation = ((np.exp(net_value)) - (np.exp(-net_value))) / ((np.exp(net_value)) + (np.exp(-net_value)))
	
	return activation


def load_data(file):
	# =============
	# path to file
	# =============
	directory = file + "/*.png"
	
	# ===============================================
	# creating a collection with the available images
	# ===============================================
	image_collection = imread_collection(directory)
	
	return image_collection


def split(data, labels, split_fraction=80):
	"""
	===============
	Parameters
	===============
	:param data: numpy array
				 data to be split into train and test
				 
	:param labels:

				 
	:param split_fraction: int
	                       fraction by which the data is to be split into train and test
	
	===============
	Return
	===============
	:return: train_data:   numpy array
	                       training data
	                
	:return: test_data:    numpy array
                           testing data
                        
    :return: train_labels: numpy array
                           ground truth of training data
                        
    :return: test_labels: numpy array
                          ground truth of testing data
	"""
	
	split_number = int((split_fraction * data.shape[0]) / 100)
	
	# =======================================
	# Splitting the data into train and test
	# =======================================
	train_data = data[0: split_number, :]
	test_data = data[split_number:, :]
	
	train_labels = labels[0: split_number]
	test_labels = labels[split_number:]
	
	return train_data, test_data, train_labels, test_labels


def load_images(file):
	"""
		This function loads that images from directory and splits the images into train and test
		
		======================================================================================
		Parameters
		======================================================================================
		:param file: string
					 directory path which contains the data

		=====================================================================================
		Return
		=====================================================================================
		:return: train_images: numpy array
		                      training image

		:return: test_images: numpy array
	                         testing image
	                         
	    :return: train_targets: numpy array
		                        training image targets
		                      
		:return: test_targets: numpy array
		                       testing image targets
		"""
	image_list = []  # List for storing all the images
	targets = []
	
	for filename in glob.glob(file + '/*.png'):
		# ==================
		# Reading the image
		# ==================
		image = scipy.misc.imread(filename).astype(np.float32)
		
		# ================================
		# Converting the image to a vector
		# ================================
		image = image.flatten()  # (784, )
		
		# ==============================
		# Normalizing the image to numpy
		# ==============================
		image = image / 255.0
		image = image - 0.5
		image = image * 2.0
		
		# ===============================
		# Appending the image to the list
		# ===============================
		image_list.append(image)
		
		_, value = filename.split('\\')
		# print(value[0])
		targets.append(int(value[0]))
	
	image_list = np.array(image_list)
	targets = np.array(targets)
	
	# ================================================
	# 			Shuffling the data
	# ================================================
	image_list, targets = shuffle(image_list, targets)
	
	train_images, test_images, train_targets, test_targets = split(image_list, targets)
	return train_images, test_images, train_targets, test_targets

# "Delta Rule"
# "Filtered Learning"
# "Unsupervised Heb"

# "Linear"
# "Hyperbolic Tangent"
# "Symmetrical Hard limit"

# model = Model("Data", "Linear", "Delta Rule")
# model.start_learning()
