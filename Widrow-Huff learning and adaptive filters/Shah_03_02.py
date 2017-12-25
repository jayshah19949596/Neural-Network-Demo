# Jai, Shah
# 1001-380-311
# 2017-10-08
# Assignment_03_02

import matplotlib
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tkinter as tk
import Shah_03_04 as s04
import Shah_03_06 as s06

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
        self.network = s04.Network(True)
        self.tf_network = s06.Network(True)
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.no_of_delay = 10
        self.learning_rate = 0.1
        self.sample_size = 80
        self.batch_size = 100
        self.iterations = 10
        #########################################################################
        #  Set up the plotting area
        #########################################################################
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, sticky=tk.N + tk.S)
        self.plot_frame.rowconfigure(0)
        self.plot_frame.columnconfigure(0)
        self.figure, self.axes_array = plt.subplots(2, 2)
        self.network.set_axes_array(self.axes_array, self.figure)
        self.figure.set_size_inches(15, 10)
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
        self.no_of_delayed_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0, to_=100, resolution=1, bg="#DDDDDD", length=200,
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Number of Delayed Elements",
                                             command=lambda event: self.no_of_delay_callback())
        self.no_of_delayed_slider.set(self.no_of_delay)
        self.no_of_delayed_slider.bind("<ButtonRelease-1>", lambda event: self.no_of_delay_callback())
        self.no_of_delayed_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
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
                                          from_=1, to_=200, resolution=1, bg="#DDDDDD", length=200,
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
        
        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################
        self.buttons_1 = tk.Button(self.sliders_frame, text="Zero Weights", fg="black",
                                   command=self.buttons_1_callback)
        self.buttons_1.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.buttons_2 = tk.Button(self.sliders_frame, text="Adjust Weights", fg="black",
                                   command=self.buttons_2_callback)
        self.buttons_2.grid(row=6, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.buttons_3 = tk.Button(self.sliders_frame, text="Adjust Weights with Tensorflow", fg="black",
                                   command=self.buttons_3_callback)
        self.buttons_3.grid(row=7, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        # self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.buttons_frame.rowconfigure(0, weight=1)
        # self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.display_activation_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())
    
    def display_activation_function(self):
        input_values = np.linspace(-10, 10, 256, endpoint=True)
        net_value = input_values
        self.axes.cla()
        self.axes.cla()
        self.axes_array[0, 0].plot(input_values, net_value)
        self.axes_array[0, 0].set_title("MSE for Price")
        self.axes_array[0, 1].plot(input_values, net_value)
        self.axes_array[0, 1].set_title("MSE for Volume")
        self.axes_array[1, 0].plot(input_values, net_value)
        self.axes_array[1, 0].set_title("MAE for Price")
        self.axes_array[1, 1].plot(input_values, net_value)
        self.axes_array[1, 1].set_title("MAE for Volume")
        self.axes.xaxis.set_visible(True)
        plt.tight_layout()
        self.canvas.draw()
    
    def no_of_delay_callback(self):
        self.no_of_delay = self.no_of_delayed_slider.get()
        self.network.set_delay(self.no_of_delay)
        self.tf_network.set_delay(self.no_of_delay)
        # self.display_activation_function()
        # print("delay", self.no_of_delay)
    
    def learning_slider_callback(self):
        self.learning_rate = self.learning_slider.get()
        self.network.set_alpha(self.learning_rate)
        self.tf_network.set_alpha(self.learning_rate)
        # self.display_activation_function()
        # print("learning_slider", self.learning_rate)
    
    def training_size_slider_callback(self):
        self.sample_size = self.training_size_slider.get()
        self.network.set_split(self.sample_size)
        self.tf_network.set_split(self.sample_size)
        # self.display_activation_function()
        # print("sample_size", self.sample_size)

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        self.network.set_batch_size(self.batch_size)
        self.tf_network.set_batch_size(self.batch_size)
        # self.display_activation_function()
        # print("batch_size", self.batch_size)

    def iteration_slider_callback(self):
        self.iterations = self.iteration_slider.get()
        self.network.set_epoch(self.iterations)
        self.tf_network.set_epoch(self.iterations)
        # self.display_activation_function()
        # print("iterations", self.iterations)
        
    def buttons_1_callback(self):
        self.network.set_weights_to_zero()
        self.tf_network.set_weights_to_zero()
        # self.display_activation_function()

    def buttons_2_callback(self):
        x_values, mse_price, mse_volume, mae_price, mae_volume = self.network.train()
        self.plot_graph(x_values, mse_price, mse_volume, mae_price, mae_volume)
        # self.display_activation_function()

    def buttons_3_callback(self):
        x_values, mse_price, mse_volume, mae_price, mae_volume = self.tf_network.training()
        self.plot_graph(x_values, mse_price, mse_volume, mae_price, mae_volume)
    
    def plot_graph(self, x_values, mse_price, mse_volume, mae_price, mae_volume):
        self.axes.cla()
        self.axes.cla()
        self.axes_array[0, 0].cla()
        self.axes_array[0, 0].plot(x_values, mse_price)
        self.axes_array[0, 0].set_title("MSE for Price")
        
        self.axes_array[0, 1].cla()
        self.axes_array[0, 1].plot(x_values, mse_volume)
        self.axes_array[0, 1].set_title("MSE for Volume")
        
        self.axes_array[1, 0].cla()
        self.axes_array[1, 0].plot(x_values, mae_price)
        self.axes_array[1, 0].set_title("MAE for Price")
        
        self.axes_array[1, 1].cla()
        self.axes_array[1, 1].plot(x_values, mae_volume)
        self.axes_array[1, 1].set_title("MAE for Volume")
        
        self.axes.xaxis.set_visible(True)
        plt.tight_layout()
        self.canvas.draw()
