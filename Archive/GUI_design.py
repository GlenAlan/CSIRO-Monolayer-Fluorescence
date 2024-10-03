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
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import typing
import threading
import queue
import math

from MCM301_COMMAND_LIB import *

from ARCHIVED_custom_definitions import *

confirmation_bits = (2147484928, 2147484930, 2147483904)
camera_dims = [2448, 2048] # This is updated dynamically later
camera_properties = {"gain": 255, "exposure": 150000}
nm_per_px = 171.6
image_overlap = 0.05
dist = int(min(camera_dims) * nm_per_px * (1-image_overlap))


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

def move_and_wait(mcm301obj, pos, stages=(4, 5)):
    """
    Moves the stage to a specified position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (tuple): The desired position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
   
    The function converts the given nanometer position into encoder units that the stage controller can use,
    then commands the stage to move to those positions. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    print(f"Moving to {', '.join(str(p) for p in pos)}")

    for i, stage in enumerate(stages):
        coord = [0]

        # Convert the positions from nanometers to encoder units
        mcm301obj.convert_nm_to_encoder(stage, pos[i], coord)

        # Move the stages to the required encoder position
        mcm301obj.move_absolute(stage, coord[0])

    moving = True
    while moving:
        moving = False
        for i, stage in enumerate(stages):
            # Wait until the stages have finished moving by checking the status bits
            bit = [0]
            mcm301obj.get_mot_status(stage, [0], bit)
            if bit[0] not in confirmation_bits:
                moving = True
            # print(bit[0])

def move_no_wait(mcm301obj, pos, stages=(4, 5)):
    """
    Moves the stage to a specified position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (tuple): The desired position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
   
    The function converts the given nanometer position into encoder units that the stage controller can use,
    then commands the stage to move to those positions. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    print(f"Moving to {', '.join(str(p) for p in pos)}")

    for i, stage in enumerate(stages):
        coord = [0]

        # Convert the positions from nanometers to encoder units
        mcm301obj.convert_nm_to_encoder(stage, pos[i], coord)

        # Move the stages to the required encoder position
        mcm301obj.move_absolute(stage, coord[0])



def get_pos(mcm301obj, stages=(4, 5, 6)):
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


def move_and_wait_relative(mcm301obj, pos=[0, 0], stages=(4, 5)):
    """
    Moves the stage to a specified position relative to the current position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (list): The desired relative position to move to, given as a list of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
   
    The function retrieves the current position of the specified stage, adds the relative position to it,
    and then moves the stage to the new position. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    pos = [p + c for p, c in zip(pos, get_pos(mcm301obj, stages))] 
    move_and_wait(mcm301obj, pos, stages)


def move_no_wait_relative(mcm301obj, pos=[0, 0], stages=(4, 5)):
    """
    Moves the stage to a specified position relative to the current position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (list): The desired relative position to move to, given as a list of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
   
    The function retrieves the current position of the specified stage, adds the relative position to it,
    and then moves the stage to the new position. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    pos = [p + c for p, c in zip(pos, get_pos(mcm301obj, stages))] 
    move_no_wait(mcm301obj, pos, stages)

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

        # Flags to control image updates
        self.update_active_main = False
        self.update_active_calib = False

        # Bind tab change event
        notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.main_frame_image = tk.Canvas(tab_main)
        self.main_frame_text = tk.Frame(tab_main, width=500, height=300, padx=50)
        self.main_frame_image.pack(side = 'left')
        self.main_frame_text.pack()

        self.camera = camera
        # Sources images from camera
        # The images are placed on a Canvas later, depending on the current tab active, to reduce lag
        self.image_acquisition_thread = ImageAcquisitionThread(self.camera, rotation_angle=90)
        self.vel = 50 # not sure what this does         
    
        # Camera parameters
        print("Setting camera parameters...")
        self.camera.frames_per_trigger_zero_for_unlimited = 0
        self.camera.arm(2)
        self.camera.issue_software_trigger()

        # Starting image acquisition thread
        print("Starting image acquisition thread...")
        self.image_acquisition_thread.start()

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

        # Results tab
        tab_results = ttk.Frame(notebook)
        notebook.add(tab_results, text="Results & Analysis")

        # Device tab
        tab_devices = ttk.Frame(notebook)
        notebook.add(tab_devices, text="Devices")
        
        self.create_control_buttons()

        self.create_sliders()

        self.create_360_wheel()

        threading.Thread(target=self.update).start()

    def on_tab_change(self, event):
        selected_tab = event.widget.index("current")
        if selected_tab == 0:  # Live View Tab 1
            self.update_active_main = True
            self.update_active_calib = False
            self.update_image_main()
        elif selected_tab == 1:  # Live View Tab 2
            self.update_active_main = False
            self.update_active_calib = True
            self.update_image_calib()
        else:
            self.update_active_main = False
            self.update_active_calib = False

    def update_image_main(self):
        if not self.update_active_main:
            return

        # Update the image on main frame
        self.calib_frame_image.delete("all")
        self.camera_widget_main = LiveViewCanvas(parent=self.main_frame_image, image_queue=self.image_acquisition_thread.get_output_queue())

        # Schedule the next update for Tab 1

    def update_image_calib(self):
        if not self.update_active_calib:
            return

        # Update the image on calibration frame
        self.calib_frame_image.delete("all")
        self.camera_widget_calib = LiveViewCanvas(parent=self.calib_frame_image, image_queue=self.image_acquisition_thread.get_output_queue())

        # Schedule the next update for Tab 2

    def update(self):
        while True:
            # Position Live view for main and calibration tabs
            for i, name in enumerate(self.pos_names):
                # Position names X,Y,Z
                label = tk.Label(self.main_frame_text_pos, text = name, padx = 10, pady = 5)
                label.grid(row = i, column = 0)
                label = tk.Label(self.calib_frame_text_pos, text = name, padx = 10, pady = 5)
                label.grid(row = i, column = 0)
            for i in range(len(self.pos_names)):
                # nm amounts
                label = tk.Label(self.main_frame_text_pos, text = f'{get_pos(self.mcm301obj, stages=(i+4,))[0]:.2e} nm', padx = 5, bg='lightgrey', width = 10)
                label.grid(row = i, column = 1)
                label = tk.Label(self.calib_frame_text_pos, text = f'{get_pos(self.mcm301obj, stages=(i+4,))[0]:.2e} nm', padx = 5, bg='lightgrey', width = 10)
                label.grid(row = i, column = 1)
            for i in range(2):
                # pixel amounts? could delete
                label = tk.Label(self.main_frame_text_pos, text = 'pixels', padx = 5, bg='lightgrey', width = 10)
                label.grid(row = i, column = 2)
                label = tk.Label(self.calib_frame_text_pos, text = 'pixels', padx = 5, bg='lightgrey', width = 10)
                label.grid(row = i, column = 2)

    def create_control_buttons(self):
        ## Main frame controls
        # List of position names and other arrays used in the tk buttons and labels
        self.pos_names = [
            "Pos X",
            "Pos Y",
            "Focus Z"
        ]

        # Label for position enterables
        self.enter_pos = tk.Label(self.main_frame_text)
        self.enter_pos.grid(row=0, pady=10)
        self.enter_focus = tk.Label(self.calib_frame_text)
        self.enter_focus.grid(row=0, pady=10)

        # Create an entry widget in main tab 
        self.pos_entry_x = tk.Entry(self.main_frame_text)
        self.pos_entry_x.grid(row=1, column=0, pady=10)
        self.pos_entry_y = tk.Entry(self.main_frame_text)
        self.pos_entry_y.grid(row=1, column=1, pady=10)
        # Entry widget for focus in calibration tab
        self.pos_entry_z = tk.Entry(self.calib_frame_text)
        self.pos_entry_z.grid(row=2, column=4, pady=10) #to update row and column

        # Bind the Enter key to the entry widget
        self.pos_entry_x.bind('<Return>', lambda event, type = "XY": self.submit_entries(type))
        self.pos_entry_y.bind('<Return>', lambda event, type = "XY": self.submit_entries(type))
        self.pos_entry_z.bind('<Return>', lambda event, type = "Z": self.submit_entries(type))

        # Create buttons in main tab and place them in the grid - origin, set start, set end required
        # for i, btn_pos in enumerate(btn_pos_nav):
        #     button = tk.Button(self.main_frame_text, text = btn_names[i], width = 22, height = 2, relief = 'groove', command = lambda: move_and_wait(self.mcm301obj, pos=btn_pos))
        #     button.grid(row = btn_positions[i][0], column = btn_positions[i][1], padx=30, pady=30) # padding around the buttons, not the text in the buttons.
        
        # Frame for positions in main
        self.main_frame_text_pos = tk.Frame(self.main_frame_text, padx = 25)
        self.main_frame_text_pos.grid(row=2, sticky='w', pady = 30) #First two rows are taken up by buttons, this frame starts at row 2 of the text frame
    
        ## Calibration frame controls
        # Z controls next to slider
        calib_button_focus_label = tk.Label(self.calib_frame_text, text="Focus Slider")
        calib_button_focus_label.grid(row=0, column=0, pady=10)

        calib_btn_up = tk.Button(self.calib_frame_text, text="Up", command = lambda: move_no_wait_relative(self.mcm301obj, pos = [int(dist/2)], stages=(5,)))
        calib_btn_up.grid(row=1, column=1)

        calib_btn_left = tk.Button(self.calib_frame_text, text="Left", command = lambda: move_no_wait_relative(self.mcm301obj, pos = [int(-dist/2)], stages=(4,)))
        calib_btn_left.grid(row=2, column=0)

        calib_btn_right = tk.Button(self.calib_frame_text, text="Right", command = lambda: move_no_wait_relative(self.mcm301obj, pos = [int(dist/2)], stages=(4,)))
        calib_btn_right.grid(row=2, column=2)

        calib_btn_down = tk.Button(self.calib_frame_text, text="Down", command = lambda: move_no_wait_relative(self.mcm301obj, pos = [int(-dist/2)], stages=(5,)))
        calib_btn_down.grid(row=3, column=1)

        calib_btn_zoom_in = tk.Button(self.calib_frame_text, text="Zoom in", command = lambda: move_no_wait_relative(self.mcm301obj, pos = [int(10000)], stages=(6,)))
        calib_btn_zoom_in.grid(row=1, column=4)

        calib_btn_zoom_out = tk.Button(self.calib_frame_text, text="Zoom out", command = lambda: move_no_wait_relative(self.mcm301obj, pos = [int(-10000)], stages=(6,)))
        calib_btn_zoom_out.grid(row=3, column=4)

    def create_sliders(self):
        # Calibration adjustment sliders. Currently there is a focus (z pos) slider. Camera rotation wheel is included elsewhere now.

        # Slider for focus (z position)
        slider_focus_label = ttk.Label(self.calib_frame_text, text="Focus Slider (Dummy Variable 1):")
        slider_focus_label.grid(row=0, column=0, pady=10)

        slider_focus = tk.Scale(self.calib_frame_text, from_=100000, to=1000000, orient='vertical', command = lambda: move_no_wait(self.mcm301obj, stages=(6,)))
        slider_focus.grid(row=0, column=1, padx=20, pady=10)
    
    def create_360_wheel(self):
        wheel_frame = ttk.Frame(self.calib_frame_text)
        wheel_frame.grid(padx=20, pady=20)

        camera_wheel_label = ttk.Label(self.calib_frame_text, text="360 Degree Wheel (Dummy Variable 2):")
        camera_wheel_label.grid(row=1, column=0, pady=10)

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

        # Camera rotation calibration
        # insert camera rotation function here # parameters: rotation = angle
        self.image_acquisition_thread._rotation_angle = angle
        print(f"Camera rotation updated to: {angle:.2f} degrees")

        # Updating the angle displayed
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{angle:.2f}")
    
    # Function to handle the submit action for X and Y
    def submit_entries_pos(self, event=None, type="XY"):
        if type == "XY":
            enter_x = self.pos_entry_x.get().strip()
            enter_y = self.pos_entry_y.get().strip()

            if not enter_x or not enter_y:
                # Show an error message if either entry is empty
                messagebox.showerror("Input Error", "Both fields are required!")
            elif not enter_x.isdigit() or not enter_y.isdigit():
                messagebox.showerror("Input Error", "Both fields must be integers!")
            else:
                # Move function when the values have been entered
                threading.Thread(target=lambda: move_and_wait(self.mcm301obj, pos=[int(enter_x)*1000000,int(enter_y)*1000000])).start()
                # # Clear the entries
                # enter_x.delete(0, tk.END)
                # enter_y.delete(0, tk.END)
        elif type == "Z":
            enter_z = self.pos_entry_z.get().strip()

            if not enter_x.isdigit() or not enter_y.isdigit():
                messagebox.showerror("Input Error", "Both fields must be integers!")
            else:
                # Move function when the values have been entered
                threading.Thread(target=lambda: move_and_wait(self.mcm301obj, pos=[int(enter_x)*1000000,int(enter_y)*1000000])).start()
    
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
        print(f"Camera rotation set to: {angle:.2f} degrees")

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