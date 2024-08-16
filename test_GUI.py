import tkinter as tk
from tkinter import ttk
import random
from PIL import Image, ImageTk

# Initialize the main window
root = tk.Tk()
root.title("Rolling a die")

# Create a Notebook (tab container)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Create the first tab
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Tab 1")

# Add content to the first tab
label1 = tk.Label(tab1, text="This is Tab 1", font=("Arial", 16))
label1.pack(pady=20)

# Trying to zoom in on cropped image
canvas = tk.Canvas(tab1, width = 1600, height = 650)
canvas.pack()
# Image handling
image_path = "Images\Screenshot 2024-07-16 154534.png"
original_image = Image.open(image_path) #Gets the image from file
tk_image = ImageTk.PhotoImage(original_image) #Creates image format or something with ImageTk
canvas.create_image(650, 0, anchor=tk.NW, image=tk_image) #Displays the image
zoom_factor = 3
region = [0, 0, 200, 200]
cropped_image = original_image.crop(region) #crops the image to the pixels desired
resized_image = cropped_image.resize(
            (int((region[2] - region[0]) * zoom_factor),
             int((region[3] - region[1]) * zoom_factor)),
            Image.LANCZOS) #resizes it to zoom into the cropped area
tk_image_zoom = ImageTk.PhotoImage(resized_image)
canvas.create_image(0,0, anchor = tk.NW, image = tk_image_zoom)

# Create a button that moves the image in x and y by 50 pixels
def move(dx=0, dy=0):
    region[0] += 50*dx
    region[2] += 50*dx
    region[1] += 50*dy
    region[3] += 50*dy
    print(region)
    cropped_image = original_image.crop(region) #crops the image to the pixels desired
    resized_image = cropped_image.resize(
                (int((region[2] - region[0]) * zoom_factor),
                int((region[3] - region[1]) * zoom_factor)),
                Image.LANCZOS) #resizes it to zoom into the cropped area - not dependent on location, just width and height
    tk_image_zoom = ImageTk.PhotoImage(resized_image)
    canvas.delete("all")
    canvas.create_image(0,0, anchor = tk.NW, image = tk_image_zoom)

button_frame = tk.Frame(tab1)
button_frame.pack(pady=10)

btn_up = tk.Button(button_frame, text="Up", command=move(dy=1))
btn_up.grid(row=0, column=1)

btn_left = tk.Button(button_frame, text="Left", command=move(dx=-1))
btn_left.grid(row=1, column=0)

btn_right = tk.Button(button_frame, text="Right", command=move(dx=1))
btn_right.grid(row=1, column=2)

btn_down = tk.Button(button_frame, text="Down", command=move(dy=-1))
btn_down.grid(row=2, column=1)

# Create the second tab
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Tab 2")

# Add content to the second tab
label2 = tk.Label(tab2, text="This is Tab 2", font=("Arial", 16))
label2.pack(pady=20)

def randNum():
    lbl_roll['text'] = random.randint(1,6)

root.columnconfigure(0, minsize=150)
root.rowconfigure([0, 1], minsize=50)

btn_roll = tk.Button(master = tab2, text = "Roll!", command=randNum)
btn_roll.pack()
lbl_roll = tk.Label(master = tab2)
lbl_roll.pack()

# Start the Tkinter event loop
root.mainloop()