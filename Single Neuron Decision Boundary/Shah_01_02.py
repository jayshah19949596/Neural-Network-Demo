# Jai, Shah
# 1001-380-311
# 2017-09-12
# Assignment_01_02

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from numpy import *
import math
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
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight_1 = 1
        self.input_weight_2 = 1
        self.bias = 0
        self.activation_function = "Linear"
        self.activation = 0
        self.data_points = None
        self.targets = None
        self.x_values = None
        self.y_values = None
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders
        
        # ================================================
        #   Setting up the weight_slider_1
        #  ===============================================
        self.weight_slider_1 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                        from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                        activebackground="#FF0000",
                                        highlightcolor="#00FFFF",
                                        label="Weight 1",
                                        command=lambda event: self.weight_slider_1_callback())
        self.weight_slider_1.set(self.input_weight_1)
        self.weight_slider_1.bind("<ButtonRelease-1>", lambda event: self.weight_slider_1_callback())
        self.weight_slider_1.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # ===============================================
        #   Setting up the weight_slider_2
        # ===============================================
        self.weight_slider_2 = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                        from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                        activebackground="#FF0000",
                                        highlightcolor="#00FFFF",
                                        label="Weight 2",
                                        command=lambda event: self.weight_slider_2_callback())
        self.weight_slider_2.set(self.input_weight_2)
        self.weight_slider_2.bind("<ButtonRelease-1>", lambda event: self.weight_slider_2_callback())
        self.weight_slider_2.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # ===============================================
        #   Setting up the bias slider
        # ===============================================
        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        
        # =================
        # Activation Label
        # =================
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        
        # =========================
        # Create Random Data Button
        # =========================
        self.buttons_1 = tk.Button(self.buttons_frame, text="Create Random Data", fg="black",
                                   command=self.create_data_callback)
        self.buttons_1.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # =========================
        # Create Train Data Button
        # =========================
        self.buttons_2 = tk.Button(self.buttons_frame, text="Train", fg="black", command=self.train_callback)
        self.buttons_2.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # ======================
        #   Setting up DropDown
        # ======================
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Linear", "Symmetrical Hard limit",
                                                          "Hyperbolic Tangent", command=lambda
                                                          event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Linear")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

    def display_activation_function(self):
        input_values_x = np.linspace(-10, 10, 256, endpoint=True)
        input_values_y = np.linspace(-10, 10, 256, endpoint=True)
        
        # =====================
        # Calculate net_value
        # =====================
        net_value = self.input_weight_1 * input_values_x + self.input_weight_2 * input_values_y + self.bias

        # =====================
        # Calculate activation
        # =====================
        self.activation = self.get_activation(net_value)

        # =====================
        # Plot the results
        # =====================
        self.axes.cla()
        self.axes.cla()
        self.axes.plot(input_values_x, self.activation)
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title(self.activation_function)
        self.canvas.draw()

    def weight_slider_1_callback(self):
        self.input_weight_1 = self.weight_slider_1.get()
        if self.x_values is not None:
            self.change_boundary()
        else:
            self.display_activation_function()
 
    def weight_slider_2_callback(self):
        self.input_weight_2 = self.weight_slider_2.get()
        if self.x_values is not None:
            self.change_boundary()
        else:
            self.display_activation_function()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        if self.x_values is not None:
            self.change_boundary()
        else:
            self.display_activation_function()

    def activation_function_dropdown_callback(self):
        self.x_values = None
        self.activation_function = self.activation_function_variable.get()
        self.display_activation_function()
        
    def change_boundary(self):
        self.x_values = np.linspace(-10, 10, 256)
    
        # ========================================================
        # Getting the y values to plot a linear decision boundary
        # ========================================================
        self.y_values = \
            ((- self.input_weight_1 * self.x_values) - self.bias) / self.input_weight_2  # Equation of a line
        
        self.plot_graph()
    
    def train_callback(self):
        """
            This function is called every time when button_2 is clicked

            This function implements the perceptron learning rule
            and shows the results of learning after each iteration on graph
        """
        self.input_weight_1 = 1.0
        self.input_weight_2 = 1.0
        self.bias = 0
        weights = np.array([self.input_weight_1, self.input_weight_2])
        outer_loop = False
        error_array = np.array([5.0, 5.0, 5.0, 5.0])
        
        # ==========================
        # Training starts from here
        # ==========================
        for i in range(0, 100):
            for j in range(0, 4):
                # =======================
                # Getting the input point
                # =======================
                point = self.data_points[j, :]
                
                # =======================
                # Calculating net value
                # =======================
                net_value = np.sum(weights * point) + self.bias  # [1x2] * [2x1]
        
                # =======================
                # Calculating error
                # =======================
                error = self.targets[j] - self.get_activation(net_value)
                error_array[j] = error
                
                # ============================================
                # Keeping the error in range from -700 to 700
                # this is to avoid nan or overflow error
                # ============================================
                if error > 1000 or error < -700:
                    error /= 10000

                # ==========================
                # Updating Weights and bias
                # ==========================
                weights += error * point
                self.bias += error * 1.0  # While updating bias input is always 1
                    
                # if weights[0] > 5000 or weights[0] < -5000:
                #     weights[0] /= 10000
                # if weights[1] > 5000 or weights[1] < -5000:
                #     weights[1] /= 10000
                # if self.bias > 5000 or self.bias < -5000:
                #     self.bias /= 10000
                
                # ==================================
                # Printing out the updated weights
                # ==================================
                # print(weights)
                
                if (error_array == np.array([0.0, 0.0, 0.0, 0.0])).all():
                    outer_loop = True
                    break
            self.x_values = np.linspace(-10, 10, 256)
            
            # ========================================================
            # Getting the y values to plot a linear decision boundary
            # ========================================================
            self.y_values = ((- weights[0] * self.x_values) - self.bias) / weights[1]  # Equation of a line
            self.input_weight_1 = weights[0]
            self.input_weight_2 = weights[1]
            
            # ================================
            # plotting the decision boundary
            # ================================
            if i % 20 == 0:
                self.plot_graph()
            
            if outer_loop:
                break
                
        self.plot_graph()
        self.input_weight_1 = weights[0]
        self.input_weight_2 = weights[1]

    def create_data_callback(self):
        """
            This function is called every time when button_1 is clicked
            
            This function will generate random data points and generate
            their target labels and plot the points on the graph
        """
        self.x_values = None
        
        # ======================================
        # generating random integer data points
        # ======================================
        self.data_points = np.random.randint(-10, 10, size=(4, 2))

        # ==================================================================
        # Defining the Target Labels for above generated random data points
        # ==================================================================
        self.targets = np.array([1.0, 1.0, -1.0, -1.0])
        self.plot_graph()

    def plot_graph(self):
        
        self.axes.cla()
        self.axes.cla()
        
        # =================================
        # Plotting the four data points
        # =================================
        self.axes.plot(self.data_points[0, 0], self.data_points[0, 1], c="black", marker="v")
        self.axes.plot(self.data_points[1, 0], self.data_points[1, 1], c="black", marker="v")
        self.axes.plot(self.data_points[2, 0], self.data_points[2, 1], c="black", marker="o")
        self.axes.plot(self.data_points[3, 0], self.data_points[3, 1], c="black", marker="o")
        
        if self.x_values is not None:
            self.axes.plot(self.x_values, self.y_values)
            
            target_pos = (self.data_points[0: 2, 1]) > (((- self.input_weight_1 * self.data_points[0: 2, 0]) - self.bias) / self.input_weight_2)
            target_neg = (self.data_points[2:, 1]) > (((- self.input_weight_1 * self.data_points[2:, 0]) - self.bias) / self.input_weight_2)
            
            # =======================================================================================
            # Below code will display the region of the input space +1 of the neuron in green color.
            # and the region of the input space -1 of the neuron  in red color.
            # =======================================================================================
            if (target_pos == np.array([True, True])).all() and (target_neg == np.array([False, False])).all():
                plt.fill_between(self.x_values, self.y_values, -10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, +10, color='green', alpha='0.75')
            elif (target_pos == np.array([False, False])).all() and (target_neg == np.array([True, True])).all():
                plt.fill_between(self.x_values, self.y_values, +10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, -10, color='green', alpha='0.75')
            elif (target_pos == np.array([False, False])).all():
                plt.fill_between(self.x_values, self.y_values, +10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, -10, color='green', alpha='0.75')
            elif (target_pos == np.array([True, True])).all():
                plt.fill_between(self.x_values, self.y_values, -10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, +10, color='green', alpha='0.75')
            elif (target_neg == np.array([True, True])).all():
                plt.fill_between(self.x_values, self.y_values, +10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, -10, color='green', alpha='0.75')
            elif (target_neg == np.array([False, False])).all():
                plt.fill_between(self.x_values, self.y_values, -10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, +10, color='green', alpha='0.75')
            else:
                plt.fill_between(self.x_values, self.y_values, -10, color='red', alpha='0.75')
                plt.fill_between(self.x_values, self.y_values, +10, color='green', alpha='0.75')

        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.title(self.activation_function)

        self.canvas.draw()

    def get_activation(self, net_value):
        if self.activation_function == 'Sigmoid':
            # =============================
            # Calculate Sigmoid Activation
            # =============================
            activation = 1.0 / (1 + np.exp(-net_value))
    
        elif self.activation_function == "Linear":
            # =============================
            # Calculate Linear Activation
            # =============================
            activation = net_value
    
        elif self.activation_function == "Symmetrical Hard limit":
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
    
        elif self.activation_function == "Hyperbolic Tangent":
            # =============================================
            # Calculate Hyperbolic Tangent Activation
            # =============================================
            activation = ((np.exp(net_value)) - (np.exp(-net_value))) / ((np.exp(net_value)) + (np.exp(-net_value)))

        return activation
