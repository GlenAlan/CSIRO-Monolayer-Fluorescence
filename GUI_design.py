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


    
""" Main

When run as a script, a simple Tkinter app is created with just a LiveViewCanvas widget. 

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