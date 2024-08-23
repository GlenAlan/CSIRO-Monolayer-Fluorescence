"""

GUI design using Tkinter with image, button controls for controlling position, and a whole tab for calibration.

"""

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import typing
import threading
import queue
import math

from MCM301_COMMAND_LIB import *

from custom_definitions import *

confirmation_bits = (2147484928, 2147484930)


def stage_setup():
    """
    Initializes and sets up the stage for movement. It first creates an instance of the MCM301 object
    and checks for connected devices. If a device is found, it connects to the first one in the list.
    The function then checks if the device is open and if not, it closes the connection and exits the script.

    After successfully opening the device, it homes the stages (in this case, stages 4 and 5).
    Homing is the process of moving the stages to a reference position. The function waits for the
    stages to complete homing by checking the status bits until they indicate that the stage is no longer moving.

    Returns:
        mcm301obj (MCM301): The initialized MCM301 object ready for further stage operations.
    """
    mcm301obj = MCM301()

    # List connected devices
    devs = MCM301.list_devices()
    print(devs)
    if len(devs) <= 0:
        print('There is no devices connected')
        exit()

    # Connect to the first available device
    device_info = devs[0]
    sn = device_info[0]
    print("connect ", sn)
    hdl = mcm301obj.open(sn, 115200, 3)
    if hdl < 0:
        print("open ", sn, " failed. hdl is ", hdl)
        exit()

    # Ensure the device is successfully opened
    if mcm301obj.is_open(sn) == 0:
        print("MCM301IsOpen failed")
        mcm301obj.close()
        exit()

    # Home the stages
    for stage_num in (4,5):
        print(f"Homing stage {stage_num}")
        mcm301obj.home(stage_num)

    # Wait for homing to complete by checking the status bits
    bits_x, bits_y = [0], [0]
    while bits_x[0] not in confirmation_bits or bits_y[0] not in confirmation_bits:
        mcm301obj.get_mot_status(4, [0], bits_x)
        mcm301obj.get_mot_status(5, [0], bits_y)
        # print(f"x: {bits_x}, y:{bits_y}")

    print("Homing complete")
    print("Stage setup complete")

    return mcm301obj

def move_and_wait(mcm301obj, pos, stage=(4, 5)):
    """
    Moves the stage to a specified position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (tuple): The desired position to move to, given as a tuple of x and y coordinates in nanometers.
        stage (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
    
    The function converts the given nanometer position into encoder units that the stage controller can use,
    then commands the stage to move to those positions. It continues to check the status of the stage 
    until it confirms that the movement is complete.
    """
    x_nm, y_nm, = pos
    print(f"Moving to {x_nm}, {y_nm}")
    x, y = [0], [0]
    stage_x, stage_y = stage
    encoder_val_x, bits_x = [0], [0]
    encoder_val_y, bits_y = [0], [0]

    # Convert the positions from nanometers to encoder units
    mcm301obj.convert_nm_to_encoder(stage_x, x_nm, x)
    mcm301obj.convert_nm_to_encoder(stage_y, y_nm, y)

    # Move the stages to the required encoder position
    mcm301obj.move_absolute(stage_x, x[0])
    mcm301obj.move_absolute(stage_y, y[0])

    # Wait until the stages have finished moving by checking the status bits
    while bits_x[0] not in confirmation_bits or bits_y[0] not in confirmation_bits:
        mcm301obj.get_mot_status(stage_x, encoder_val_x, bits_x)
        mcm301obj.get_mot_status(stage_y, encoder_val_y, bits_y)
        # print(f"x: {bits_x}, y:{bits_y}")

def get_pos(mcm301obj, stages=[4, 5, 6]):
    """
    Retrieves the current position of the specified stages.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        stages (list): A list of stage numbers for which the positions are to be retrieved.
    
    Returns:
        pos (list): A list of positions corresponding to the specified stages, 
                    in nanometers.
    
    The function queries the current encoder value for each specified stage, 
    converts that value into nanometers, and returns the positions as a list.
    """
    pos = []
    for stage in stages:
        encoder_val, nm = [0], [0]

        # Get the current encoder value for the stage
        mcm301obj.get_mot_status(stage, encoder_val, [0])

        # Convert the encoder value to nanometers
        mcm301obj.convert_encoder_to_nm(stage, encoder_val[0], nm)

        # Append the position to the list
        pos.append(nm[0])
    return pos

class GUI:
    def __init__(self, root, camera):
        self.root = root
        self.root.title('Camera with controls - development')
        
        # Create a Notebook (tab container)
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")

        # Creating a tab for main then with frames below. The code is organised by tabs.
        tab_main = ttk.Frame(notebook)
        notebook.add(tab_main, text="Main Tab")

        self.main_frame_image = tk.Canvas(tab_main)
        self.main_frame_text = tk.Frame(tab_main, width=500, height=300, padx=50)
        self.main_frame_image.pack(side = 'left')
        self.main_frame_text.pack()

        self.camera = camera
        # Sources images from camera and places it on canvas
        image_acquisition_thread = ImageAcquisitionThread(self.camera)
        self.camera_widget_main = LiveViewCanvas(parent=self.main_frame_image, image_queue=image_acquisition_thread.get_output_queue())
        self.vel = 50 # not sure what this does         
    
        # Camera parameters
        print("Setting camera parameters...")
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.arm(2)
        self.camera.issue_software_trigger()

        # Starting image acquisition thread
        print("Starting image acquisition thread...")
        image_acquisition_thread.start()

        # Initialises object
        self.mcm301obj = stage_setup()

        # Calibration tab
        tab_calibrate = ttk.Frame(notebook)
        notebook.add(tab_calibrate, text="Calibration")

        self.calib_frame_image = tk.Canvas(tab_calibrate)
        self.calib_frame_image.pack(side = 'left')
        self.calib_frame_text = tk.Frame(tab_calibrate, width=500, height=300, padx=50)
        self.calib_frame_text.pack()
        
        self.calib_frame_text_pos = tk.Frame(self.calib_frame_text, padx = 25)
        self.calib_frame_text_pos.grid(row=2 , sticky='w', pady = 30) #First two rows are taken up by buttons, this frame starts at row 2 of the text frame
        
        # Live image view in calibration tab
        self.camera_widget_calib = LiveViewCanvas(parent=self.calib_frame_image, image_queue=image_acquisition_thread.get_output_queue())

        # Results tab
        tab_results = ttk.Frame(notebook)
        notebook.add(tab_results, text="Results & Analysis")

        # Device tab
        tab_devices = ttk.Frame(notebook)
        notebook.add(tab_devices, text="Devices")
        
        self.create_control_buttons()

        self.create_sliders()

        self.create_360_wheel()
        
        self.update()

    def update(self):
        # Positions - Live view of X,Y, and Z (focus) required
        for i, name in enumerate(self.pos_names):
            label = tk.Label(self.main_frame_text_pos, text = name, padx = 10, pady = 5)
            label.grid(row = i, column = 0)
            label = tk.Label(self.calib_frame_text_pos, text = name, padx = 10, pady = 5)
            label.grid(row = i, column = 0)
        for i in range(len(self.pos_names)):
            label = tk.Label(self.main_frame_text_pos, text = f'{get_pos(self.mcm301obj, stages=(i+4,))[0]:.2e} nm', padx = 5, bg='lightgrey', width = 10)
            label.grid(row = i, column = 1)
            label = tk.Label(self.calib_frame_text_pos, text = f'{get_pos(self.mcm301obj, stages=(i+4,))[0]:.2e} nm', padx = 5, bg='lightgrey', width = 10)
            label.grid(row = i, column = 1)
        for i in range(2):
            label = tk.Label(self.main_frame_text_pos, text = 'pixels', padx = 5, bg='lightgrey', width = 10)
            label.grid(row = i, column = 2)
            label = tk.Label(self.calib_frame_text_pos, text = 'pixels', padx = 5, bg='lightgrey', width = 10)
            label.grid(row = i, column = 2)
        root.after(100, self.update)

    def create_control_buttons(self):
        # List of button names, positions, and other arrays used in the tk buttons and labels
        btn_pos_nav = [
            (5000000,5000000),
            (1000500,1000500)
        ]
        btn_names = [
            f"Move to {btn_pos_nav[0]}",
            f"Move to {btn_pos_nav[1]}",
            "Zoom in \'some amount\'",
            "Zoom out \'some amount\'",
        ]
        btn_positions = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.pos_names = [
            "Pos X",
            "Pos Y",
            "Focus Z"
        ]

        # Create buttons in main tab and place them in the grid - origin, set start, set end required
        for i, btn_pos in enumerate(btn_pos_nav):
            button = tk.Button(self.main_frame_text, text = btn_names[i], width = 22, height = 2, relief = 'groove', command = lambda: move_and_wait(self.mcm301obj, pos=btn_pos))
            button.grid(row = btn_positions[i][0], column = btn_positions[i][1], padx=30, pady=30) # padding around the buttons, not the text in the buttons.
        
        # Frame for positions in main
        self.main_frame_text_pos = tk.Frame(self.main_frame_text, padx = 25)
        self.main_frame_text_pos.grid(row=2, sticky='w', pady = 30) #First two rows are taken up by buttons, this frame starts at row 2 of the text frame
    
    def create_sliders(self):
        # Calibration adjustment sliders. Currently there is a focus (z pos) slider. To include camer rotation wheel.
        # Sliders do not have variable or command assigned to it yet.
        # First slider needs z position function. Second slider needs camera rotation function.

        # Slider 1
        slider_focus_label = ttk.Label(self.calib_frame_text, text="Dummy Variable 1:")
        slider_focus_label.grid(row=0, column=0, pady=10)

        slider_focus = tk.Scale(self.calib_frame_text, from_=0, to=100, orient='vertical',)
        slider_focus.grid(row=0, column=1, padx=20, pady=10)
    
    def create_360_wheel(self):
        wheel_frame = ttk.Frame(self.calib_frame_text)
        wheel_frame.grid(padx=20, pady=20)

        camera_wheel_label = ttk.Label(self.calib_frame_text, text="360 Degree Wheel (Dummy Variable 2):")
        camera_wheel_label.pack(row=1, column=0, pady=10)

        # Create a canvas for the 360-degree wheel
        self.canvas = tk.Canvas(wheel_frame, width=200, height=200, bg="white")
        self.canvas.pack()

        # Set up image parameters
        self.radius = 80  # Radius of the wheel
        self.center_x = 100  # Center X of the wheel
        self.center_y = 100  # Center Y of the wheel

        # Create a blank image with transparency
        self.image = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.image)

        # Draw the anti-aliased wheel circle
        self.draw_anti_aliased_wheel()

        # Convert the image to Tkinter-compatible format
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Display the wheel on the canvas
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Draw the initial pointer
        self.pointer_line = self.canvas.create_line(self.center_x, self.center_y,
                                                    self.center_x + self.radius, self.center_y,
                                                    width=2, fill="red")

        # Bind mouse events to drag the wheel
        self.canvas.bind("<B1-Motion>", self.update_wheel)

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
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

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
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        # Update the dummy variable 2
        self.dummy_var2.set(angle)
        print(f"Dummy Variable 2 set to: {angle:.2f} degrees")

"""
Main. Runs the main code that create the root window from tkinter and everything inside it.
"""
if __name__ == "__main__":
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        with sdk.open_camera(camera_list[0]) as camera:

            # create generic Tk App with just a LiveViewCanvas widget
            print("App initialising...")
            root = tk.Tk()
            GUI_main = GUI(root, camera)
            
            print("App starting")
            root.mainloop()

            print("Waiting for image acquisition thread to finish...")
            GUI_main.image_acquisition_thread.stop()
            GUI_main.image_acquisition_thread.join()

            print("Closing resources...")

    print("App terminated. Goodbye!")