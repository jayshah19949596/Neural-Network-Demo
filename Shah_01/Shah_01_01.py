# Jai, Shah
# 1001-380-311
# 2017-09-12
# Assignment_01_01

import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import tensorflow as tf
import Shah_01_02 as s02  # This module is for plotting components
import Shah_01_03 as s03  # This module is for graphic components


class WidgetsWindow(tk.Tk):
    def __init__(self):
        print(tf.__version__)
        tk.Tk.__init__(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.menu_bar = MenuBar(self, self, background='orange')
        self.tool_bar = ToolBar(self, self)
        # self.status_bar = StatusBar(self, self, bg='red', bd=1, relief=tk.SUNKEN)

        self.center_frame = tk.Frame(self)
        # Create a frame for plotting graphs
        self.left_frame = PlotsDisplayFrame(self, self.center_frame, bg='blue')
        self.display_activation_functions = s02.DisplayActivationFunctions(self, self.left_frame)
        # Create a frame for displaying graphics
        # self.right_frame = GraphicsDisplayFrame(self, self.center_frame, bg='yellow')
        # self.display_graphics = s03.DisplayGraphics(self, self.right_frame)
        self.tool_bar.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.tool_bar.rowconfigure(0, minsize=20)
        self.center_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.center_frame.grid_propagate(True)
        self.center_frame.rowconfigure(1, weight=1, uniform='xx')
        self.center_frame.columnconfigure(0, weight=1, uniform='xx')
        # self.center_frame.columnconfigure(1, weight=1, uniform='xx')
        # self.status_bar.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.status_bar.rowconfigure(2, minsize=20)
        self.left_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # self.right_frame.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)


class MenuBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.menu = tk.Menu(self.root)
        root.config(menu=self.menu)
        self.filemenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="New", command=self.menu_callback)
        self.filemenu.add_command(label="Open...", command=self.menu_callback)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.menu_callback)
        self.dummymenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Dummy", menu=self.dummymenu)
        self.dummymenu.add_command(label="Item1", command=self.menu_item1_callback)
        self.dummymenu.add_command(label="Item2", command=self.menu_item2_callback)
        self.helpmenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=self.helpmenu)
        self.helpmenu.add_command(label="About...", command=self.menu_help_callback)

    def menu_callback(self):
        self.root.status_bar.set('%s', "called the menu callback!")

    def menu_help_callback(self):
        self.root.status_bar.set('%s', "called the help menu callback!")

    def menu_item1_callback(self):
        self.root.status_bar.set('%s', "called item1 callback!")

    def menu_item2_callback(self):
        self.root.status_bar.set('%s', "called item2 callback!")


class ToolBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.master = master

        self.draw_button = tk.Button(self, text="Draw", fg="red", width=16, command=self.toolbar_draw_callback)
        self.draw_button.grid(row=0, column=0)
        self.var_filename = tk.StringVar()
        self.var_filename.set('')

        self.ask_for_string = tk.Button(self, text="Ask for a string", command=self.ask_for_string)
        self.ask_for_string.grid(row=0, column=1)

        self.file_dialog_button = tk.Button(self, text="Open File Dialog", fg="blue", command=self.browse_file)
        self.file_dialog_button.grid(row=0, column=2)

        self.open_dialog_button = tk.Button(self, text="Open Dialog", fg="blue", command=self.open_dialog_callback)
        self.open_dialog_button.grid(row=0, column=3)

    def say_hi(self):
        self.root.status_bar.set('%s', "hi there, everyone!")

    def ask_for_string(self):
        s = simpledialog.askstring('My Dialog', 'Please enter a string')
        self.root.status_bar.set('%s', s)

    def ask_for_float(self):
        f = float(simpledialog.askfloat('My Dialog', 'Please enter a float'))

        self.root.status_bar.set('%s', str(f))

    def browse_file(self):
        self.var_filename.set(tk.filedialog.askopenfilename(filetypes=[("allfiles", "*"), ("pythonfiles", "*.txt")]))
        filename = self.var_filename.get()
        self.root.status_bar.set('%s', filename)

    def open_dialog_callback(self):
        d = MyDialog(self.root)
        self.root.status_bar.set('%s', "my_dialog_callback pressed. Returned results: " + str(d.result))

    def button2_callback(self):
        self.root.status_bar.set('%s', 'button2 pressed.')

    def toolbar_draw_callback(self):
        self.root.display_graphics.create_graphic_objects()
        self.root.status_bar.set('%s', "called the draw callback!")

    def toolbar_callback(self):
        self.root.status_bar.set('%s', "called the toolbar callback!")
        

class MyDialog(tk.simpledialog.Dialog):
    def body(self, parent):

        tk.Label(parent, text="Integer:").grid(row=0, sticky=tk.W)
        tk.Label(parent, text="Float:").grid(row=1, column=0, sticky=tk.W)
        tk.Label(parent, text="String:").grid(row=1, column=2, sticky=tk.W)
        self.e1 = tk.Entry(parent)
        self.e1.insert(0, 0)
        self.e2 = tk.Entry(parent)
        self.e2.insert(0, 4.2)
        self.e3 = tk.Entry(parent)
        self.e3.insert(0, 'Default text')

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=1, column=3)

        self.cb = tk.Checkbutton(parent, text="Hardcopy")
        self.cb.grid(row=3, columnspan=2, sticky=tk.W)

    def apply(self):
        try:
            first = int(self.e1.get())
            second = float(self.e2.get())
            third = self.e3.get()
            self.result = first, second, third
        except ValueError:
            tk.tkMessageBox.showwarning(
                "Bad input",
                "Illegal values, please try again"
            )


class StatusBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.label = tk.Label(self)
        self.label.grid(row=0, sticky=tk.N + tk.E + tk.S + tk.W)

    def set(self, set_format, *args):
        self.label.config(text=set_format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()


class PlotsDisplayFrame(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        # self.configure(width=500, height=500)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.bind("<Key>", self.key_pressed_callback)
        self.bind("<Up>", self.up_arrow_pressed_callback)
        self.bind("<Down>", self.down_arrow_pressed_callback)
        self.bind("<Right>", self.right_arrow_pressed_callback)
        self.bind("<Left>", self.left_arrow_pressed_callback)
        self.bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.bind("f", self.f_key_pressed_callback)
        self.bind("b", self.b_key_pressed_callback)

    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was released. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse button was released. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.focus_set()


class GraphicsDisplayFrame(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.master=master
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg='yellow')
        self.canvas.rowconfigure(0, weight=1)
        self.canvas.columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.canvas.bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.canvas.bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.canvas.bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.canvas.bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.canvas.bind("<Key>", self.key_pressed_callback)
        self.canvas.bind("<Up>", self.up_arrow_pressed_callback)
        self.canvas.bind("<Down>", self.down_arrow_pressed_callback)
        self.canvas.bind("<Right>", self.right_arrow_pressed_callback)
        self.canvas.bind("<Left>", self.left_arrow_pressed_callback)
        self.canvas.bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.canvas.bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.canvas.bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.canvas.bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.canvas.bind("f", self.f_key_pressed_callback)
        self.canvas.bind("b", self.b_key_pressed_callback)

    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was released. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse button was released. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' +
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.focus_set()
    
    @staticmethod
    def frame_resized_callback(event):
        print("frame resize callback")


# z = WidgetsWindow()
# z.mainloop()

def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

widgets_window = WidgetsWindow()
# widgets_window.geometry("500x500")
# widgets_window.wm_state('zoomed')
widgets_window.title('Assignment_00 -- Shah')
widgets_window.minsize(600, 300)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()
