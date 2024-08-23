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
import typing
import threading
import queue

from MCM301_COMMAND_LIB import *

confirmation_bits = (2147484928, 2147484930)


""" LiveViewCanvas

This is a Tkinter Canvas object that can be reused in custom programs. The Canvas expects a parent Tkinter object and 
an image queue. The image queue is a queue.Queue that it will pull images from, and is expected to hold PIL Image 
objects that will be displayed to the canvas. It automatically adjusts its size based on the incoming image dimensions.

"""


class LiveViewCanvas(tk.Canvas):

    def __init__(self, parent, image_queue):
        # type: (typing.Any, queue.Queue) -> LiveViewCanvas
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0

        tk.Canvas.__init__(self, parent)
        self.grid(column=0, row=0, rowspan=4, columnspan=2, sticky=tk.E)
        self._get_image()

    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            
            # resize image
            image = image.resize((500,500))
            
            self._image = ImageTk.PhotoImage(master=self, image=image)
            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                # resize the canvas to match the new image size
                self._image_width = 500 #self._image.width()
                self._image_height = 500 #self._image.height()
                self.config(width=self._image_width, height=self._image_height)
            self.create_image(0, 0, image=self._image, anchor='nw')
        except queue.Empty:
            pass
        self.after(10, self._get_image)


""" ImageAcquisitionThread

This class derives from threading.Thread and is given a TLCamera instance during initialization. When started, the 
thread continuously acquires frames from the camera and converts them to PIL Image objects. These are placed in a 
queue.Queue object that can be retrieved using get_output_queue(). The thread doesn't do any arming or triggering, 
so users will still need to setup and control the camera from a different thread. Be sure to call stop() when it is 
time for the thread to stop.

"""


class ImageAcquisitionThread(threading.Thread):

    def __init__(self, camera):
        # type: (TLCamera) -> ImageAcquisitionThread
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._previous_timestamp = 0

        # setup color processing if necessary
        if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            # Sensor type is not compatible with the color processing library
            self._is_color = False
        else:
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._image_width = self._camera.image_width_pixels
            self._image_height = self._camera.image_height_pixels
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER,
                self._camera.color_filter_array_phase,
                self._camera.get_color_correction_matrix(),
                self._camera.get_default_white_balance_matrix(),
                self._camera.bit_depth
            )
            self._is_color = True

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0  # Do not want to block for long periods of time
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_color_image(self, frame):
        # type: (Frame) -> Image
        # verify the image size
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]
        if (width != self._image_width) or (height != self._image_height):
            self._image_width = width
            self._image_height = height
            print("Image dimension change detected, image acquisition thread was updated")
        # color the image. transform_to_24 will scale to 8 bits per channel
        color_image_data = self._mono_to_color_processor.transform_to_24(frame.image_buffer,
                                                                         self._image_width,
                                                                         self._image_height)
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)
        # return PIL Image object
        return Image.fromarray(color_image_data, mode='RGB')

    def _get_image(self, frame):
        # type: (Frame) -> Image
        # no coloring, just scale down image to 8 bpp and place into PIL Image object
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        return Image.fromarray(scaled_image)

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    if self._is_color:
                        pil_image = self._get_color_image(frame)
                    else:
                        pil_image = self._get_image(frame)
                    self._image_queue.put_nowait(pil_image)
            except queue.Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        print("Image acquisition has stopped")
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()


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
Main. Runs the main code that create the root window from tkinter and 
"""
if __name__ == "__main__":
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        with sdk.open_camera(camera_list[0]) as camera:

            # create generic Tk App with just a LiveViewCanvas widget
            print("App initialising...")
            root = tk.Tk()
            root.title('Camera with controls - development')

            # Create a Notebook (tab container)
            notebook = ttk.Notebook(root)
            notebook.pack(expand=True, fill="both")

            # Creating a tab for main. The code is organised by tabs
            tab_main = ttk.Frame(notebook)
            notebook.add(tab_main, text="Main Tab")

            # Creating the frames for the main tab. Note: this does not include anything inside the frames yet.
            frame_image = tk.Canvas(tab_main)
            frame_text = tk.Frame(tab_main, width=500, height=300, padx=50)
            frame_image.pack(side = 'left')
            frame_text.pack()

            # Sources images from camera and places it on canvas
            image_acquisition_thread = ImageAcquisitionThread(camera)
            camera_widget = LiveViewCanvas(parent=frame_image, image_queue=image_acquisition_thread.get_output_queue())
            vel = 50 # not sure what this does         
        
            # See if this can be skipped
            print("Setting camera parameters...")
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.arm(2)
            camera.issue_software_trigger()

            # Not sure when image acquisition thread needs to start in the code.
            print("Starting image acquisition thread...")
            image_acquisition_thread.start()

            # Initialises object
            mcm301obj = stage_setup()

            frame_pos = tk.Frame(frame_text, padx = 25)
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
            pos_names = [
                "Pos X",
                "Pos Y",
                "Focus Z"
            ]

            # Create buttons and place them in the grid
            # Buttons required - origin, set start, set end
            for i, btn_pos in enumerate(btn_pos_nav):
                button = tk.Button(frame_text, text = btn_names[i], width = 22, height = 2, relief = 'groove', command = lambda: move_and_wait(mcm301obj, pos=btn_pos))
                button.grid(row = btn_positions[i][0], column = btn_positions[i][1], padx=30, pady=30) # padding around the buttons, not the text in the buttons.
            
            def update():
                # Positions
                # Positions required - Live view of X,Y, and Z (focus)
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

            # Need to check when to call update()
            update()

            
            print("App starting")
            root.mainloop()

            print("Waiting for image acquisition thread to finish...")
            image_acquisition_thread.stop()
            image_acquisition_thread.join()

            print("Closing resources...")

    print("App terminated. Goodbye!")