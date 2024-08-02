# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:57:06 2024

@author: Admin
"""

import numpy as np
import time

# Run Setup File
runfile('C:/Users/Admin/Documents/Programming/MCM301 SKD/Stage_Controller.py', wdir='C:/Users/Admin/Documents/Programming/MCM301 SKD')

# Define Coordinate System
x_start = 10
x_width = 10
x_num = 4

y_start = 10
y_width = 10
y_num = 4

# Define stage parameters
vel = 50

x_pos = np.arange(x_start, x_start+x_width, x_width/x_num)
y_pos = np.arange(y_start, y_start+y_width, y_width/y_num)

for x in x_pos:
    for y in y_pos:
        move(4, x, vel)
        move(5, y, vel)
        print("At position (", x, ",", y, ")")
        time.sleep(2)
        
        
        
        
        