# Jai, Shah
# 1001-380-311
# 2017-09-27
# Assignment_02_02

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from numpy import *
import math
import Shah_02_04 as s04
matplotlib.use('TkAgg')

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
        self.xmin = 0
        self.xmax = 1000
        self.ymin = -0.1
        self.ymax = 1.0
        self.alpha = 0.1
        self.input_weight_2 = 1
        self.alpha_value = 0
        self.learning_method = "Filtered Learning"
        self.activation_function = "Linear"
        self.activation_variable = "Linear"
        self.activation = 0
        self.data_points = None
        self.targets = None
        self.x_values = None
        self.y_values = None
        self.model = None
        self.display = None
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.btn_and_slider_frame = tk.Frame(self.master)
        self.btn_and_slider_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.btn_and_slider_frame.rowconfigure(0, weight=10)
        self.btn_and_slider_frame.rowconfigure(1, weight=2)
        self.btn_and_slider_frame.columnconfigure(0, weight=5, uniform='xx')
        self.btn_and_slider_frame.columnconfigure(1, weight=1, uniform='xx')
        
        # ================================================
        #   Setting up the alpha_slider for learning rate
        #  ===============================================
        self.alpha_slider = tk.Scale(self.btn_and_slider_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                     from_=0.001, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                     activebackground="#FF0000",
                                     highlightcolor="#00FFFF",
                                     label="Learning Rate",
                                     command=lambda event: self.alpha_slider_callback())
        
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################
        
        self.drop_down_frame = tk.Frame(self.master)
        self.drop_down_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.drop_down_frame.rowconfigure(0, weight=1)
        self.drop_down_frame.columnconfigure(0, weight=1, uniform='xx')
        
        # ======================
        # Learning Method Label
        # ======================
        self.label_for_learning_method = tk.Label(self.drop_down_frame, text="Learning Method",
                                                  justify="center")
        self.label_for_learning_method.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.learning_method_variable = tk.StringVar()
        
        self.label_for_activation_function = tk.Label(self.drop_down_frame, text="Activation",
                                                      justify="center")
        self.label_for_activation_function.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_variable = tk.StringVar()
        
        # =============================
        # Create Adjust Weights Button
        # =============================
        self.buttons_1 = tk.Button(self.btn_and_slider_frame, text="Adjust Weights(Learn)", fg="black",
                                   command=self.adjust_weights_callback)
        self.buttons_1.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # ================================
        # Create Randomize Weights Button
        # ================================
        self.buttons_2 = tk.Button(self.btn_and_slider_frame, text="Randomize Weights", fg="black",
                                   command=self.randomize_weights_callback)
        self.buttons_2.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # ======================
        # Setting up DropDown
        # ======================
        self.learning_method_drop_down = tk.OptionMenu(self.drop_down_frame, self.learning_method_variable,
                                                       "Filtered Learning", "Delta Rule",
                                                       "Unsupervised Heb", command=lambda
                event: self.learning_method_callback())
        self.learning_method_variable.set("Filtered Learning")
        self.learning_method_drop_down.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.activation_drop_down = tk.OptionMenu(self.drop_down_frame, self.activation_variable,
                                                  "Linear", "Symmetrical Hard limit",
                                                  "Hyperbolic Tangent", command=lambda
                event: self.activation_function_dropdown_callback())
        self.activation_variable.set("Linear")
        self.activation_drop_down.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())
        self.display_activation_function()
    
    def alpha_slider_callback(self):
        self.alpha_value = self.alpha_slider.get()
    
    def learning_method_callback(self):
        self.x_values = None
        self.learning_method = self.learning_method_variable.get()
        self.display_activation_function()
    
    def activation_function_dropdown_callback(self):
        self.x_values = None
        self.activation_function = self.activation_variable.get()
        self.display_activation_function()
    
    def randomize_weights_callback(self):
        """
            This function is called every time when button_2 is clicked

            This function implements the perceptron learning rule
            and shows the results of learning after each iteration on graph
        """
        
        self.model = s04.Model("Data", self.activation_function, self.learning_method, self.alpha_value)
        self.display_activation_function()
    
    def adjust_weights_callback(self):
        """
            This function is called every time when button_1 is clicked
            
            This function will generate random data points and generate
            their target labels and plot the points on the graph
        """
        self.display_activation_function()
        self.model.start_learning()
        self.plot_error(self.model.epochs, self.model.error)
    
    def display_activation_function(self):
        self.axes.cla()
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error')
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title("Activation = "+self.activation_function+"\nLearning Method = "+self.learning_method)
        self.canvas.draw()
    
    def plot_error(self, epoch, error):
        self.axes.cla()
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        self.axes.set_xlabel('Epochs')
        self.axes.set_ylabel('Error')
        self.axes.plot(epoch, error)
        self.xmax = len(epoch)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title("Activation = "+self.activation_function+"\nLearning Method = "+self.learning_method)
        self.canvas.draw()
        if len(epoch) == 1000:
            self.randomize_weights_callback()

