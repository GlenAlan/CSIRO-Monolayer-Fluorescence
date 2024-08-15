"""
Camera Live View - TkInter

This example shows how one could create a live image viewer using TkInter.
It also uses the third party library 'pillow', which is a fork of PIL.

This example detects if a camera is a color camera and will process the
images using the tl_mono_to_color_processor module.

This example uses threading to enqueue images coming off the camera in one thread, and
dequeue them in the UI thread for quick displaying.

"""

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

# from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
# from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
# from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import typing
import threading
import queue
    
""" Main

GUI design with image, button controls for controlling position, and a whole tab for calibration.

"""
if __name__ == "__main__":
    # The following is opening the camera which is not used in this case.
    # with TLCameraSDK() as sdk:
    #     camera_list = sdk.discover_available_cameras()
    #     with sdk.open_camera(camera_list[0]) as camera:

    root = tk.Tk() #Create an instance of a window
    root.title('Camera with controls - development')

    image_path = "CSIRO-Monolayer-Fluorescence\Images\Screenshot 2024-07-16 154534.png"
    #gets the image in a ImageTK type
    img = ImageTk.PhotoImage(Image.open(image_path).resize((400,400))) #Need to check if resize affects calibration/measurement.
    stitched_img = ImageTk.PhotoImage(Image.open(image_path).resize((400,400)))

    # Create a Notebook (tab container)
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    # Creating the tabs
    tab_main = ttk.Frame(notebook)
    notebook.add(tab_main, text="Main Tab")
    tab_calibrate = ttk.Frame(notebook)
    notebook.add(tab_calibrate, text="Calibration Tab")

    # Creating the frames in the tabs
    frame_image = tk.Frame(master = tab_main)
    frame_text = tk.Frame(master = tab_main, width=500, height=300, padx=50)
    frame_image.pack(side = 'left')
    frame_text.pack()

    frame_pos = tk.Frame(master = frame_text, padx = 25)
    frame_pos.grid(row=2, sticky='w', pady = 30) #First two rows are taken up by buttons, this frame starts at row 2 of the text frame
    
    # List of button names
    btn_names = [
        "Move left \'some amount\'",
        "Move right \'some amount\'",
        "Zoom out \'some amount\'",
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
        "Pos Y"
    ]
    positions = [
        '100nm',
        '200nm'
    ]

    image = tk.Label(master = frame_image, image = img) #Creates a Label wiget
    image.pack()
    image_result = tk.Label(master = frame_image, image = stitched_img)
    image_result.pack()

    # Create buttons and place them in the grid
    # Buttons required - origin, set start, set end
    for i in range(len(btn_names)):
        button = tk.Button(frame_text, text = btn_names[i], width = 22, height = 2, relief = 'groove') #, command = command_list
        button.grid(row = btn_positions[i][0], column = btn_positions[i][1], padx=30, pady=30) #padding around the buttons, not the text in the buttons.
    
    # Positions
    for i in range(len(pos_names)):
        label = tk.Label(frame_pos, text = pos_names[i], padx = 10, pady = 5)
        label.grid(row = i, column = 0)
    for i in range(len(positions)):
        label = tk.Label(frame_pos, text = positions[i], padx = 5, bg='lightgrey', width = 10)
        label.grid(row = i, column = 1)
    
    # Create a calibration tab
    
    #Runs the window application mainloop
    root.mainloop()
    
    # # create generic Tk App with just a LiveViewCanvas widget
    # print("Generating app...")
    # root = tk.Tk()
    # root.title('Trying to view an image in a window')
    # camera_widget = LiveViewCanvas(parent=root, image_queue=image_acquisition_thread.get_output_queue())
    # vel = 50      

    # # create generic Tk App with just a LiveViewCanvas widget
    # print("Generating app...")
    # root = tk.Tk()
    # root.title(camera.name)
    # image_acquisition_thread = ImageAcquisitionThread(camera)
    # camera_widget = LiveViewCanvas(parent=root, image_queue=image_acquisition_thread.get_output_queue())
    # vel = 50            

    # print("Setting camera parameters...")
    # camera.frames_per_trigger_zero_for_unlimited = 0
    # camera.arm(2)
    # camera.issue_software_trigger()

    # print("Starting image acquisition thread...")
    # image_acquisition_thread.start()

    # print("App starting")
    # root.mainloop()

    # print("Waiting for image acquisition thread to finish...")
    # image_acquisition_thread.stop()
    # image_acquisition_thread.join()

    # print("Closing resources...")

    # print("App terminated. Goodbye!")