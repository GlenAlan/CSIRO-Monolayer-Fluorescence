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
from PIL import Image, ImageTk

from MCM301_COMMAND_LIB import *

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

    
""" 
Main
"""
if __name__ == "__main__":
    # Create an instance of a window
    root = tk.Tk()
    root.title('Camera with controls - development')

    mcm301obj = stage_setup()

    # Create a Notebook (tab container)
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    # Creating a tab for main. The code is organised by tabs
    tab_main = ttk.Frame(notebook)
    notebook.add(tab_main, text="Main Tab")

    # Creating the frames for the main tab. Note: this does not include anything inside the frames yet.
    frame_image = tk.Canvas(master = tab_main)
    frame_text = tk.Frame(master = tab_main, width=500, height=300, padx=50)
    frame_image.pack(side = 'left')
    frame_text.pack()

    frame_pos = tk.Frame(master = frame_text, padx = 25)
    frame_pos.grid(row=2, sticky='w', pady = 30) #First two rows are taken up by buttons, this frame starts at row 2 of the text frame
    
    # List of button names, positions, and other arrays used in the tk buttons and labels
    btn_pos_nav = [
        (5000,5000),
        (10500,10500)
    ]
    btn_names = [
        f"Move to {btn_pos_nav[0]}",
        f"Move to {btn_pos_nav[1]}",
        "Zoom in \'some amount\'",
        "Zoom out \'some amount\'",
    ]
    btn_positions = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # command_list = [
    #     insert_move_left_command
    #     insert_move_right_command
    #     insert_command_3
    #     insert_command_4
    # ]
    pos_names = [
        "Pos X",
        "Pos Y",
        "Focus Z"
    ]

    # Image handling
    image_path = "Images\Screenshot 2024-07-16 154534.png"
    # Gets the image in a ImageTK type
    img = ImageTk.PhotoImage(Image.open(image_path).resize((400,400))) #Need to check if resize affects calibration/measurement.
    stitched_img = ImageTk.PhotoImage(Image.open(image_path).resize((400,400)))

    # image = tk.Label(master = frame_image, image = img) #Creates a Label wiget
    # image.resize((int(image.width * 1.2), int(image.height * 1.2)))
    # image.pack() #packs the Label widget into the frame
    image_result = tk.Label(frame_image, image = stitched_img)
    image_result.pack()

    # Create buttons and place them in the grid
    # Buttons required - origin, set start, set end
    for i, btn_pos in enumerate(btn_pos_nav):
        button = tk.Button(frame_text, text = btn_names[i], width = 22, height = 2, relief = 'groove', command = move_and_wait(mcm301obj, pos=btn_pos)) #, command = command_list[1]
        button.grid(row = btn_positions[i][0], column = btn_positions[i][1], padx=30, pady=30) # padding around the buttons, not the text in the buttons.
    
    # Positions
    # Positions required - Live view of X,Y, and Z (focus)

    def update():
        for i, name in enumerate(pos_names):
            label = tk.Label(frame_pos, text = name, padx = 10, pady = 5)
            label.grid(row = i, column = 0)
        for i in range(len(pos_names)):
            label = tk.Label(frame_pos, text = f'{get_pos(mcm301obj, stages=(i+4,))[0]:.2e} nm', padx = 5, bg='lightgrey', width = 10)
            label.grid(row = i, column = 1)
        for i in range(2):
            label = tk.Label(frame_pos, text = 'pixels', padx = 5, bg='lightgrey', width = 10)
            label.grid(row = i, column = 2)
        root.after(100, update)
    

    # Calibration tab
    tab_calibrate = ttk.Frame(notebook)
    notebook.add(tab_calibrate, text="Calibration")

    # Results tab
    tab_results = ttk.Frame(notebook)
    notebook.add(tab_results, text="Results & Analysis")

    # Device tab
    tab_devices = ttk.Frame(notebook)
    notebook.add(tab_devices, text="Devices")

    update()
    
    # Runs the window application mainloop
    root.mainloop()