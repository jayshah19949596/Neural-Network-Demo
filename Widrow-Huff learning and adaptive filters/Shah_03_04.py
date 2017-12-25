# Jai, Shah
# 1001-380-311
# 2017-10-08
# Assignment_03_04

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math


class Network(object):
	
	def __init__(self, is_bias):
		self.split = 80
		self.train_data, self.test_data = load_data('stock_data.csv', self.split)
		self.delay = 10
		self.batch_size = 100
		self.is_bias = is_bias
		# self.weights = np.random.uniform(-0.001, 0.001, (2, (2*(self.delay+1))+1))
		
		if self.is_bias:
			self.weights = np.zeros((2, (2*(self.delay+1))+1))
		else:
			self.weights = np.zeros((2, (2 * (self.delay + 1))))
			
		self.alpha = 0.1
		self.epoch = 10
		self.figure, self.axes_array = plt.subplots(2, 2)
		self.figure.set_size_inches(5, 5)
		self.axes = self.figure.gca()
		
	def train(self):
		itr_list = []
		mse_price = []
		mse_volume = []
		mae_price = []
		mae_volume = []

		itr = 0
		for e in range(0, self.epoch):
			for i in range(0, self.train_data.shape[0], self.batch_size):
				train_batch = self.train_data[i:i + self.batch_size, :]
				data = 0
				for j in range(0, train_batch.shape[0] - (self.delay+2)):
					data += 1
					
					train_delay = np.concatenate((train_batch[j: j+self.delay+1, 0], train_batch[j: j+self.delay+1, 1]))
					if self.is_bias:
						train_delay = np.concatenate((train_delay, np.array([1])))
					train_delay = train_delay.reshape((train_delay.shape[0], 1))
					
					target = train_batch[j + self.delay + 2, :]
					target = target.reshape((train_batch.shape[1], 1))
					
					output = self.weights.dot(train_delay)
					np.nan_to_num(output)
					
					error = target - output
					np.nan_to_num(error)
					
					product_1 = 2 * self.alpha * error
					self.weights += product_1.dot(np.transpose(train_delay))
					np.nan_to_num(self.weights)
				
				itr_list.append(itr)
				
				w, x, y, z = self.testing()
				
				mse_price.append(w)
				mse_volume.append(x)
				mae_price.append(y)
				mae_volume.append(z)
				itr += 1
				
		# plot_graph(itr_list, mse_price, mse_volume, mae_price, mae_volume, self.axes_array)
		return itr_list, mse_price, mse_volume, mae_price, mae_volume
	
	def test(self, itr):
		print(itr)
		total = 0
		total_error = np.array([[0.0], [0.0]])
		outer_loop = 0
		test_data = 0
		for p in range(0, self.test_data.shape[0], self.batch_size):
			outer_loop += 1
			test_batch = self.test_data[p:p + self.batch_size, :]
			for q in range(0, test_batch.shape[0] - self.delay+2):
				test_data += 1
				total += 1
				test_delay = np.concatenate(
						(test_batch[q: q + self.delay + 1, 0], test_batch[q: q + self.delay + 1, 1]))
				if self.is_bias:
					test_delay = np.concatenate((test_delay, np.array([1])))
				 
				test_delay = test_delay.reshape((test_delay.shape[0], 1))
				
				target = test_batch[q + self.delay + 2, :]
				target = target.reshape((test_batch.shape[1], 1))
				
				output = self.weights.dot(test_delay)
				np.nan_to_num(output)
				
				error = target - output
				error = np.square(error)
				
				total_error += error
				
		total_error = total_error/test_data
		return total_error[0, 0], total_error[1, 0]
		
	def testing(self):
		total_error = 0
		maximum_abs_price = -1
		maximum_abs_volume = -1
		for p in range(0, self.test_data.shape[0] - (self.delay+2)):
			input_vector = np.concatenate(
					(self.test_data[p: p + self.delay + 1, 0], self.test_data[p: p + self.delay + 1, 1]))
			if self.is_bias:
				input_vector = np.concatenate((input_vector, np.array([1])))
			input_vector = input_vector.reshape((input_vector.shape[0], 1))
			
			target = self.test_data[p + self.delay + 2, :]
			target = target.reshape((self.test_data.shape[1], 1))
			
			output = self.weights.dot(input_vector)
			np.nan_to_num(output)
			
			error = target - output
			np.nan_to_num(error)
			
			maximum_abs_price = max(maximum_abs_price, abs(error[0, 0]))
			maximum_abs_volume = max(maximum_abs_volume, abs(error[1, 0]))

			error = np.square(error)
			np.nan_to_num(error)
			
			total_error += error
			np.nan_to_num(total_error)
			
		total_error = total_error / self.test_data.shape[0]
		print("mse price", total_error[0, 0])
		print("mse volume", total_error[1, 0])
		print("mae price", maximum_abs_price)
		print("mae volume", maximum_abs_volume)
		print("===================================")
		return total_error[0, 0], total_error[1, 0], maximum_abs_price, maximum_abs_volume
	
	def set_delay(self, delay):
		self.delay = delay
		if self.is_bias:
			self.weights = np.zeros((2, (2*(self.delay+1))+1))
		else:
			self.weights = np.zeros((2, (2 * (self.delay + 1))))
		
	def set_alpha(self, alpha):
		self.alpha = alpha
		
	def set_split(self, training_size):
		self.split = training_size
		self.train_data, self.test_data = load_data('stock_data.csv', self.split)

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		
	def set_epoch(self, epoch):
		self.epoch = epoch
		
	def set_weights_to_zero(self):
		if self.is_bias:
			self.weights = np.zeros((2, (2*(self.delay+1))+1))
		else:
			self.weights = np.zeros((2, (2 * (self.delay + 1))))
		
	def set_axes_array(self, axes_array, figure):
		self.axes_array = axes_array
		self.figure = figure
		

def plot_graph(x_axis, mse_price, mse_volume, mae_price, mae_volume, axes_array):
	axes_array[0, 0].plot(x_axis, mse_price)
	axes_array[0, 0].set_title("MSE for Price")
	axes_array[0, 1].plot(x_axis, mse_volume)
	axes_array[0, 1].set_title("MSE for Volume")
	axes_array[1, 0].plot(x_axis, mae_price)
	axes_array[1, 0].set_title("MAE for Price")
	axes_array[1, 1].plot(x_axis, mae_volume)
	axes_array[1, 1].set_title("MAE for Volume")
	plt.tight_layout()
	plt.show()


def split(data, split_fraction=80):
	split_number = int((split_fraction * data.shape[0]) / 100)
	train = data[0: split_number, :]
	test = data[split_number:, :]
	return train, test


def normalization(data):
	for i in range(0, data.shape[1]):
		maximum = np.max(data[0:, i])
		data[0:, i] /= maximum
	data -= 0.5
	return data


def load_data(file, split_fraction):
	data = np.loadtxt(file, skiprows=1, delimiter=',', dtype=np.float64)
	data = normalization(data)
	train, test = split(data, split_fraction)
	return train, test


if __name__ == "__main__":
	network = Network(True)
	network.train()

