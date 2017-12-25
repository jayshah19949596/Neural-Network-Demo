# Jai, Shah
# 1001-380-311
# 2017-10-20
# Assignment_04_04

import tensorflow as tf
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist.train._images = np.vstack((mnist.train._images, mnist.test._images))
mnist.train._labels = np.vstack((mnist.train._labels, mnist.test._labels))

epochs = 100
split_fraction = 10
transfer_function = "Relu"
hidden_nodes = 100
batch_size = 64
learning_rate = 0.1
beta = 0.01
cost_function = "Cross Entropy"
output_function = "Softmax"
start = 0
total_epochs = 0
error_list = []
epoch_list = []
loss_list = []
sess = None

with tf.variable_scope("weights_scope_1"):
	f0_w = tf.get_variable("weights_0", initializer=tf.random_normal([784, hidden_nodes], stddev=0.1))
	f0_b = tf.get_variable("bias_0", initializer=tf.random_normal([hidden_nodes]))
	f1_w = tf.get_variable("weights_1", initializer=tf.random_normal([hidden_nodes, 10], stddev=0.1))
	f1_b = tf.get_variable("bias_1", initializer=tf.random_normal([10]))


def my_model(data):
	regularizer = tf.contrib.layers.l2_regularizer(beta)

	with tf.variable_scope("weights_scope_1", reuse=True):
		fi_w = tf.get_variable("weights_0")
		fi_b = tf.get_variable("bias_0")
	f0 = tf.add(tf.matmul(data, fi_w), fi_b)

	if transfer_function == "Relu":
		f0 = tf.nn.relu(f0)
	else:
		f0 = tf.nn.sigmoid(f0)

	with tf.variable_scope("weights_scope_1", reuse=True):
		fh_w = tf.get_variable("weights_1")
		fh_b = tf.get_variable("bias_1")
	op = tf.matmul(f0, fh_w) + fh_b

	penalty = tf.contrib.layers.apply_regularization(regularizer, weights_list=[fi_w, fh_w])

	if cost_function == "Cross Entropy":
		return op, penalty

	if output_function == "Softmax":
		return tf.nn.softmax(op), penalty
	elif output_function == "Sigmoid":
		return tf.nn.sigmoid(op), penalty


def evaluate_model(x, prediction, y):
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	test_size = int(mnist.train.num_examples - split())
	test_data, test_label = mnist.train.next_batch(test_size)

	test_data = test_data - 0.5
	test_data = test_data * 2.0

	return accuracy.eval({x: test_data, y: test_label}), test_data, test_label


def split():
	split_number = int(((100-split_fraction) * mnist.train._images.shape[0]) / 100)
	return split_number


def train_neural_network(s_02, reuse):
	global total_epochs
	global start
	global error_list
	global epoch_list
	global loss_list
	global sess
	mnist.train._num_examples = mnist.train._images.shape[0]

	total_epochs += 1

	x = tf.placeholder('float', [None, 784])
	y = tf.placeholder('float')

	prediction, penalty = my_model(x)

	if cost_function == "Cross Entropy":
		if output_function == "Softmax":
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
		elif output_function == "Sigmoid":
			cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
	elif cost_function == "MSE":
		cost = tf.reduce_mean(tf.square(y - prediction))

	cost += beta*penalty

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	with sess.as_default():

		if not reuse:
			sess.run(tf.global_variables_initializer())

		for epoch in tqdm(range(start, total_epochs*10)):
			epoch_loss = 0

			for _ in range(int(split()/batch_size)):

				batch_x, batch_y = mnist.train.next_batch(batch_size)
				mnist.train.next_batch(batch_size)

				batch_x = batch_x - 0.5
				batch_x = batch_x * 2.0

				_, res = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += res

			acc, test_data, test_labels = evaluate_model(x, prediction, y)

			error_list.append(1-acc)
			epoch_list.append(epoch)
			loss_list.append(epoch_loss)

			if (epoch % 2) == 0 and epoch % 10 is not 0:
				plt.cla()
				plt.clf()

				plt.subplot(221)
				plt.plot(epoch_list, error_list)
				plt.title('Error Graph')

				plt.subplot(222)
				plt.plot(epoch_list, loss_list)
				plt.title('Loss Graph')

				s_02.canvas.draw()

		predictions = sess.run(tf.argmax(prediction, 1), feed_dict={x: test_data})
		decoded = sess.run(tf.argmax(test_labels, axis=1))

		cm = calculate_confusion_matrix(decoded, predictions)

	start = total_epochs*10

	return error_list, loss_list, epoch_list, cm


def calculate_confusion_matrix(y_test, y_prediction):
	cnf_matrix = confusion_matrix(y_test, y_prediction)
	np.set_printoptions(precision=2)
	return cnf_matrix


def plot_graph(x, y, z):
	figure, axes_array = plt.subplots(2, 1)
	axes_array[0].plot(z, x)
	axes_array[0].set_title("Error Graph")
	axes_array[1].plot(z, y)
	axes_array[1].set_title("Loss Graph")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	epochs = 100
	split_fraction = 80
	transfer_function = "Relu"
	hidden_nodes = 100
	batch_size = 64
	learning_rate = 0.1
	beta = 0.01
	cost_function = "Cross Entropy"
	output_function = "Softmax"
	iteration = 0
	total_epochs = 0
	error_list = []
	epoch_list = []
	loss_list = []
	sess = tf.Session()
	train_neural_network(False)
