import tkinter as tk
from tkinter import ttk, filedialog
import random
from PIL import Image, ImageTk


class ImageDisplay:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Rolling a die")

        # Create a Notebook (tab container)
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")

        # Create the first tab
        self.tab1 = ttk.Frame(notebook)
        notebook.add(self.tab1, text="Tab 1")

        # Add content to the first tab
        label1 = tk.Label(self.tab1, text="This is Tab 1", font=("Arial", 16))
        label1.pack(pady=20)
        # Create the second tab
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="Tab 2")

        # Add content to the second tab
        label2 = tk.Label(tab2, text="This is Tab 2", font=("Arial", 16))
        label2.pack(pady=20)

        self.root.columnconfigure(0, minsize=150)
        self.root.rowconfigure([0, 1], minsize=50)

        self.btn_roll = tk.Button(tab2, text = "Roll!", command = self.randNum)
        self.btn_roll.pack()
        self.lbl_roll = tk.Label(tab2)
        self.lbl_roll.pack()

        # Trying to zoom in on cropped image - First creating a canvas
        self.canvas = tk.Canvas(self.tab1, width = 1400, height = 600)
        self.canvas.pack()
        # Image handling
        self.image_path = "Images\Screenshot 2024-07-16 154534.png"
        self.original_image = Image.open(self.image_path) #Gets the image from file
        # Initial factors
        self.zoom_factor = 3
        self.region = [0, 0, 200, 200]

        self.update_image()

        self.create_control_buttons()

    '''
    Random number generator
    '''
    def randNum(self):
        self.lbl_roll['text'] = random.randint(1,6)
    
    '''
    Updates the zoomed image. This updates the region of the image and displays it again.
    It is called once at the start and then whenever the move function is called.
    '''
    def update_image(self):
        self.tk_image = ImageTk.PhotoImage(self.original_image) #Creates image format or something with ImageTk
        self.canvas.create_image(650, 0, anchor=tk.NW, image=self.tk_image) #Displays the image
        cropped_image = self.original_image.crop(self.region) #crops the image to the pixels desired
        resized_image = cropped_image.resize(
                    (int((self.region[2] - self.region[0]) * self.zoom_factor),
                    int((self.region[3] - self.region[1]) * self.zoom_factor)),
                    Image.LANCZOS) #resizes it to zoom into the cropped area - not dependent on location, just width and height
        self.tk_image_zoom = ImageTk.PhotoImage(resized_image)
        # canvas.delete("all") #could have something to do with needing to delete specific image before creating it.
        self.canvas.create_image(0, 0, anchor = tk.NW, image = self.tk_image_zoom)
        print('Test to see if function works')

    # Create a button that moves the image in x and y by 50 pixels
    def move(self, dx=0, dy=0):
        self.region[0] += 50*dx
        self.region[2] += 50*dx
        self.region[1] += 50*dy
        self.region[3] += 50*dy
        print(self.region)
        self.update_image()

    def create_control_buttons(self):
        button_frame = tk.Frame(self.tab1)
        button_frame.pack(pady=10)

        self.btn_up = tk.Button(button_frame, text="Up", command = lambda: self.move(dy=-1))
        self.btn_up.grid(row=0, column=1)

        btn_left = tk.Button(button_frame, text="Left", command = lambda: self.move(dx=-1))
        btn_left.grid(row=1, column=0)

        btn_right = tk.Button(button_frame, text="Right", command = lambda: self.move(dx=1))
        btn_right.grid(row=1, column=2)

        btn_down = tk.Button(button_frame, text="Down", command = lambda: self.move(dy=1))
        btn_down.grid(row=2, column=1)

# Start the Tkinter event loop
if __name__ == "__main__":
    root = tk.Tk()
    ImageDisplay(root)
    root.mainloop()