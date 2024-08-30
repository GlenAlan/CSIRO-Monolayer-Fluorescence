import tkinter as tk
from tkinter import ttk, filedialog
import random
import math
from PIL import Image, ImageDraw, ImageTk

'''
Class for running tkinter root
'''
class ImageDisplay:
    '''
    Initialising for creating tabs, labels and initial images.
    '''
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Image display and rolling a die")

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
        self.tab2 = ttk.Frame(notebook)
        notebook.add(self.tab2, text="Tab 2")

        # Create the third tab
        self.tab3 = ttk.Frame(notebook)
        notebook.add(self.tab3, text="Tab 3")

        # Add content to the second tab
        label2 = tk.Label(self.tab2, text="This is Tab 2", font=("Arial", 16))
        label2.pack(pady=20)

        self.root.columnconfigure(0, minsize=150)
        self.root.rowconfigure([0, 1], minsize=50)

        self.btn_roll = tk.Button(self.tab2, text = "Roll!", command = self.randNum)
        self.btn_roll.pack()
        self.lbl_roll = tk.Label(self.tab2)
        self.lbl_roll.pack()

        # Create an entry widget
        self.entry = tk.Entry(self.root)
        self.entry.pack(pady=10)

        # Bind the Enter key to the entry widget
        self.entry.bind('<Return>', self.get_entry_value)

        # Trying to zoom in on cropped image - First creating a canvas
        self.image_canvas = tk.Canvas(self.tab1, width = 1400, height = 600)
        self.image_canvas.pack()
        # Image handling
        self.image_path = "Images/test_image4.jpg"
        self.original_image = Image.open(self.image_path) #Gets the image from file
        # Initial factors
        self.zoom_factor = 3
        self.region = [0, 0, 200, 200]

        # Dummy variables
        self.dummy_var1 = tk.DoubleVar(value=0)
        self.dummy_var2 = tk.DoubleVar(value=0)

        self.update_image()

        self.create_control_buttons()

        self.create_sliders()

        self.create_360_wheel()

    '''
    Random number generator for die.
    '''
    def randNum(self):
        self.lbl_roll['text'] = random.randint(1,6)
    
    '''
    Updates the zoomed image. This updates the region of the image and displays it again.
    It is called once at the start and then whenever the move function is called.
    '''
    def update_image(self):
        self.tk_image = ImageTk.PhotoImage(self.original_image) #Creates image format or something with ImageTk
        self.image_canvas.create_image(650, 0, anchor=tk.NW, image=self.tk_image) #Displays the image
        cropped_image = self.original_image.crop(self.region) #crops the image to the pixels desired
        resized_image = cropped_image.resize(
                    (int((self.region[2] - self.region[0]) * self.zoom_factor),
                    int((self.region[3] - self.region[1]) * self.zoom_factor)),
                    Image.LANCZOS) #resizes it to zoom into the cropped area - not dependent on location, just width and height
        self.tk_image_zoom = ImageTk.PhotoImage(resized_image)
        self.image_canvas.create_image(0, 0, anchor = tk.NW, image = self.tk_image_zoom)

    '''
    Movement function for button that moves the image in x and y by 50 pixels
    '''
    def move(self, dx=0, dy=0):
        self.region[0] += 50*dx
        self.region[2] += 50*dx
        self.region[1] += 50*dy
        self.region[3] += 50*dy
        print(self.region)
        self.update_image()

    '''
    Creates the buttons
    '''
    def create_control_buttons(self):
        button_frame = tk.Frame(self.tab1)
        button_frame.pack(pady=10)

        btn_up = tk.Button(button_frame, text="Up", command = lambda: self.move(dy=-1))
        btn_up.grid(row=0, column=1)

        btn_left = tk.Button(button_frame, text="Left", command = lambda: self.move(dx=-1))
        btn_left.grid(row=1, column=0)

        btn_right = tk.Button(button_frame, text="Right", command = lambda: self.move(dx=1))
        btn_right.grid(row=1, column=2)

        btn_down = tk.Button(button_frame, text="Down", command = lambda: self.move(dy=1))
        btn_down.grid(row=2, column=1)

    def create_sliders(self):
        # Slider 1 for Dummy Variable 1
        slider1_label = ttk.Label(self.tab3, text="Dummy Variable 1:")
        slider1_label.grid(row=0, column=0, pady=10)

        # Slider displays value from variable and does command whenever you slide it.
        # Could use command to update z position to change focus and/or have position as the variable.
        # Could also just not have variable displayed and have it as a label somewhere else.
        slider1 = tk.Scale(self.tab3, from_=0, to=100, orient='vertical')
        slider1.grid(row=0, column=1, padx=20, pady=10)

        # Slider 2 for Dummy Variable 2
        slider2_label = ttk.Label(self.tab3, text="Dummy Variable 2:")
        slider2_label.grid(row=1, column=0, pady=10)

        slider2 = tk.Scale(self.tab3, from_=0, to=100, orient='vertical',
                           variable=self.dummy_var2, command=self.update_var2)
        slider2.grid(row=1, column=1, padx=20, pady=10)

    def create_360_wheel(self):
        wheel_frame = ttk.Frame(self.tab3)
        wheel_frame.grid(padx=20, pady=20)

        wheel_label = ttk.Label(wheel_frame, text="360 Degree Wheel (Dummy Variable 2):")
        wheel_label.pack()

        # Create a canvas for the 360-degree wheel
        self.wheel_canvas = tk.Canvas(wheel_frame, width=200, height=200, bg="white")
        self.wheel_canvas.pack()

        # Set up image parameters
        self.radius = 80  # Radius of the wheel
        self.center_x = 100  # Center X of the wheel
        self.center_y = 100  # Center Y of the wheel

        # Create a blank image with transparency
        self.wheel_image = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.wheel_image)

        # Draw the anti-aliased wheel circle
        self.draw_anti_aliased_wheel()

        # Convert the image to Tkinter-compatible format
        self.tk_wheel_image = ImageTk.PhotoImage(self.wheel_image)

        # Display the wheel on the canvas
        self.wheel_circle = self.wheel_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_wheel_image)

        # Draw the initial pointer
        self.pointer_line = self.wheel_canvas.create_line(self.center_x, self.center_y,
                                                    self.center_x + self.radius, self.center_y,
                                                    width=2, fill="red")

        # Bind mouse events to drag the wheel
        self.wheel_canvas.bind("<B1-Motion>", self.update_wheel)

        # Entry box for manually setting the angle
        self.angle_entry = tk.Entry(wheel_frame, width=5)
        self.angle_entry.pack(pady=10)
        self.angle_entry.insert(0, "0")
        self.angle_entry.bind("<Return>", self.set_angle_from_entry)

    def draw_anti_aliased_wheel(self):
        # Draw the circle for the wheel with anti-aliasing
        self.draw.ellipse(
            [self.center_x - self.radius, self.center_y - self.radius,
             self.center_x + self.radius, self.center_y + self.radius],
            outline="black", width=2
        )

    def update_wheel(self, event):
        # Calculate angle from center to current mouse position
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        angle = math.degrees(math.atan2(dy, dx)) % 360

        # Update the pointer line position
        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.wheel_canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)


        # Update the dummy variable 2
        self.dummy_var2.set(angle)
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{angle:.2f}")
        print(f"Dummy Variable 2 updated to: {angle:.2f} degrees")
    
    def set_angle_from_entry(self, event):
        # Get the angle from the entry box
        try:
            angle = float(self.angle_entry.get()) % 360
        except ValueError:
            angle = 0

        # Update the pointer line based on the entered angle
        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.wheel_canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        # Update the dummy variable 2
        self.dummy_var2.set(angle)
        print(f"Dummy Variable 2 set to: {angle:.2f} degrees")
    
    def update_var1(self, value):
        print(f"Dummy Variable 1 updated to: {value}")

    def update_var2(self, value):
        print(f"Dummy Variable 2 updated to: {value}")

    # Create a function to get the entry value
    def get_entry_value(self, event=None):
        entered_text = self.entry.get()
        print(f"You entered: {entered_text}")

# Start the Tkinter event loop
if __name__ == "__main__":
    root = tk.Tk()
    ImageDisplay(root)
    root.mainloop()