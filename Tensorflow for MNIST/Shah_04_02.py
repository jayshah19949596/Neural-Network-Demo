# Jai, Shah
# 1001-380-311
# 2017-10-20
# Assignment_04_02

import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tkinter as tk
import Shah_04_04 as s04
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class DisplayActivationFunctions:
    """
	This class is for displaying activation functions for NN.
	Farhad Kamangar 2017_08_26
	"""
    
    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.no_of_nodes = 100
        self.learning_rate = 0.1
        self.sample_size = 10
        self.batch_size = 64
        self.iterations = 100
        self.beta = 0.01
        self.activation_function = "Relu"
        self.activation_variable = "Relu"
        self.cost_function = "Softmax"
        self.cost_function_variable = "Cross Entropy"
        self.op_activation = "Softmax"
        self.op_activation_variable = "Softmax"
        self.reuse = False
        self.error_list = []
        self.loss_list = []
        self.epoch_list = []
        s04.sess = tf.Session()
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.sliders_frame.rowconfigure(0, weight=1)
        # self.sliders_frame.rowconfigure(1, weight=1)
        # self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        # self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        self.no_of_nodes_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                           from_=0, to_=500, resolution=1, bg="#DDDDDD", length=200,
                                           activebackground="#FF0000",
                                           highlightcolor="#00FFFF",
                                           label="Number of Hidden Nodes",
                                           command=lambda event: self.no_of_nodes_callback())
        self.no_of_nodes_slider.set(self.no_of_nodes)
        self.no_of_nodes_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_nodes_callback())
        self.no_of_nodes_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.learning_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                        from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD", length=200,
                                        activebackground="#FF0000",
                                        highlightcolor="#00FFFF",
                                        label="Learning Rate",
                                        command=lambda event: self.learning_slider_callback())
        self.learning_slider.set(self.learning_rate)
        self.learning_slider.bind("<ButtonRelease-1>", lambda event: self.learning_slider_callback())
        self.learning_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.training_size_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0, to_=100, resolution=1, bg="#DDDDDD", length=200,
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Training Size",
                                             command=lambda event: self.training_size_slider_callback())
        self.training_size_slider.set(self.sample_size)
        self.training_size_slider.bind("<ButtonRelease-1>", lambda event: self.training_size_slider_callback())
        self.training_size_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.batch_size_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                          from_=1, to_=256, resolution=1, bg="#DDDDDD", length=200,
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF",
                                          label="Batch Size",
                                          command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.iteration_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                         from_=1, to_=100, resolution=1, bg="#DDDDDD", length=200,
                                         activebackground="#FF0000",
                                         highlightcolor="#00FFFF",
                                         label="Iterations",
                                         command=lambda event: self.iteration_slider_callback())
        self.iteration_slider.set(self.iterations)
        self.iteration_slider.bind("<ButtonRelease-1>", lambda event: self.iteration_slider_callback())
        self.iteration_slider.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.lambda_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=0, to_=1.0, resolution=0.01, bg="#DDDDDD", length=200,
                                      activebackground="#FF0000",
                                      highlightcolor="#00FFFF",
                                      label="Lambda",
                                      command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.beta)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################
        self.activation_label = tk.Label(self.sliders_frame, text="transfer function for hidden layer",
                                         justify="center")
        self.activation_label.grid(row=6, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_variable = tk.StringVar()
        
        self.activation_drop_down = tk.OptionMenu(self.sliders_frame, self.activation_variable,
                                                  "Relu", "Sigmoid",
                                                  command=lambda
                                                      event: self.activation_function_dropdown_callback())
        self.activation_variable.set("Relu")
        self.activation_drop_down.grid(row=7, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.op_activation_label = tk.Label(self.sliders_frame, text="transfer function for output layer",
                                            justify="center")
        self.op_activation_label.grid(row=8, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.op_activation_variable = tk.StringVar()
        
        self.op_activation_drop_down = tk.OptionMenu(self.sliders_frame, self.op_activation_variable,
                                                     "Softmax", "Sigmoid",
                                                     command=lambda
                                                         event: self.op_activation_drop_down_callback())
        self.op_activation_variable.set("Softmax")
        self.op_activation_drop_down.grid(row=9, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.cost_function_label = tk.Label(self.sliders_frame, text="Cost function",
                                            justify="center")
        self.cost_function_label.grid(row=10, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.cost_function_variable = tk.StringVar()
        
        self.cost_function_drop_down = tk.OptionMenu(self.sliders_frame, self.cost_function_variable,
                                                     "Cross Entropy", "MSE",
                                                     command=lambda
                                                         event: self.cost_function_drop_down_callback())
        self.cost_function_variable.set("Cross Entropy")
        self.cost_function_drop_down.grid(row=11, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.buttons_1 = tk.Button(self.sliders_frame, text="Zero Weights", fg="black",
                                   command=self.buttons_1_callback)
        self.buttons_1.grid(row=12, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.buttons_2 = tk.Button(self.sliders_frame, text="Adjust Weights", fg="black",
                                   command=self.buttons_2_callback)
        self.buttons_2.grid(row=13, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())
    
    def display_activation_function(self):
        plt.tight_layout()
        self.canvas.draw()
    
    def no_of_nodes_callback(self):
        self.no_of_nodes = self.no_of_nodes_slider.get()
        s04.hidden_nodes = self.no_of_nodes
        print("hidden_nodes", s04.hidden_nodes)
    
    def learning_slider_callback(self):
        self.learning_rate = self.learning_slider.get()
        s04.learning_rate = self.learning_rate
        print("learning_rate", s04.learning_rate)
    
    def training_size_slider_callback(self):
        self.sample_size = self.training_size_slider.get()
        s04.split_fraction = self.sample_size
        print("split_fraction", s04.split_fraction)
    
    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        s04.batch_size = self.batch_size
        print("batch_size", s04.batch_size)
    
    def iteration_slider_callback(self):
        self.iterations = self.iteration_slider.get()
        s04.epochs = self.iterations
        print("epochs", s04.epochs)
    
    def lambda_slider_callback(self):
        self.beta = self.lambda_slider.get()
        s04.beta = self.beta
        print("beta", s04.beta)
    
    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_variable.get()
        s04.transfer_function = self.activation_function
        print("transfer_function", s04.transfer_function)
        self.display_activation_function()
    
    def cost_function_drop_down_callback(self):
        self.cost_function = self.cost_function_variable.get()
        s04.cost_function = self.cost_function
        print("cost_function", s04.cost_function)
    
    def op_activation_drop_down_callback(self):
        self.op_activation = self.op_activation_variable.get()
        s04.output_function = self.op_activation
        print("output_function", s04.output_function)
    
    def buttons_1_callback(self):
        self.reuse = False
        re_initialize()
    
    def buttons_2_callback(self):
        x, y, z, cm = s04.train_neural_network(self, self.reuse)
        
        if len(x) == 100:
            self.reuse = False
            re_initialize()
        else:
            self.reuse = True
            
        self.plot_graph(x, y, z, cm)
    
    def plot_graph(self, x, y, z, cm):
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cmap = plt.cm.Blues
        
        self.axes.cla()
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        
        plt.cla()
        plt.clf()
        
        plt.subplot(221)
        plt.plot(z, x)
        plt.title('Error Graph')
        
        plt.subplot(222)
        plt.plot(z, y)
        plt.title('Loss Graph')
        plt.rcParams["figure.figsize"] = (20, 20)
        
        plt.subplot(212)
        plt.title('Confusion Matrix')
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        normalize = False
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # print("i", i)
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.tight_layout()
        
        self.canvas.draw()


def re_initialize():
    s04.sess = tf.Session()
    s04.start = 0
    s04.total_epochs = 0
    s04.error_list = []
    s04.epoch_list = []
    s04.loss_list = []
