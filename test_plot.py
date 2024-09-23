import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np

# Create the main Tkinter window
root = tk.Tk()
root.title("Interactive Matplotlib Plot in Tkinter")

# Create a Matplotlib figure and axes
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)

# Generate some example data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Plot the data on the axes
ax.plot(x, y)

# Add the Matplotlib figure to a Tkinter canvas widget
canvas = FigureCanvasTkAgg(fig, master=root)  # A Tkinter widget containing the Matplotlib figure
canvas.draw()

# Add the canvas to the Tkinter window
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create a toolbar for interactivity (zoom, pan, save, etc.)
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Optional: Add a button to quit the application
quit_button = tk.Button(master=root, text="Quit", command=root.quit)
quit_button.pack(side=tk.BOTTOM)

# Start the Tkinter event loop
root.mainloop()
