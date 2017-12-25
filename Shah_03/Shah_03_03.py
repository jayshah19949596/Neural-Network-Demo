# Jai, Shah
# 1001-380-311
# 2017-10-08
# Assignment_03_03

import numpy as np
import tkinter as tk


class DisplayGraphics:
    def __init__(self, root, master, objects=[]):
        self.root = root
        self.master = master
        self.objects = objects
        self.canvas = self.master.canvas

    def create_graphic_objects(self):
        self.objects.append(self.canvas.create_oval(int(0.25 * int(self.canvas.cget("width"))),
                                               int(0.25 * int(self.canvas.cget("height"))),
                                               int(0.75 * int(self.canvas.cget("width"))),
                                               int(0.75 * int(self.canvas.cget("height")))))
        self.objects.append(self.canvas.create_oval(int(0.30 * int(self.canvas.cget("width"))),
                                               int(0.30 * int(self.canvas.cget("height"))),
                                               int(0.70 * int(self.canvas.cget("width"))),
                                               int(0.70 * int(self.canvas.cget("height")))))
        self.objects.append(self.canvas.create_oval(int(0.35 * int(self.canvas.cget("width"))),
                                               int(0.35 * int(self.canvas.cget("height"))),
                                               int(0.65 * int(self.canvas.cget("width"))),
                                               int(0.65 * int(self.canvas.cget("height")))))
    
    def redisplay(self, event):
        if self.objects:
            self.canvas.coords(self.objects[0], int(0.25 * int(event.width)),
                          int(0.25 * int(event.height)),
                          int(0.75 * int(event.width)),
                          int(0.75 * int(event.height)))
            self.canvas.coords(self.objects[1], int(0.30 * int(event.width)),
                          int(0.30 * int(event.height)),
                          int(0.70 * int(event.width)),
                          int(0.70 * int(event.height)))
            self.canvas.coords(self.objects[2], int(0.35 * int(event.width)),
                          int(0.35 * int(event.height)),
                          int(0.65 * int(event.width)),
                          int(0.65 * int(event.height)))

