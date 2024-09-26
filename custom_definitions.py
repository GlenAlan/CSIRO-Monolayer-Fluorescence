import os
import time
import random
import math
import typing
import threading
import queue
import operator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw

import cv2  # OpenCV library for accessing the webcam
import numpy as np
from skimage.measure import shannon_entropy
import torch

from MCM301_COMMAND_LIB import *
import config

# Thorlabs TSI SDK imports
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame, TLCameraError
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

# Windows-specific setup
try:
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

def image_to_stage(center_coords, start, factor=1):
    x, y = factor*center_coords[0], factor*center_coords[1] 
    center_coords_raw = int((x - config.CAMERA_DIMS[0])* config.NM_PER_PX + start[0]), int((y - config.CAMERA_DIMS[1])* config.NM_PER_PX + start[1])
    return center_coords_raw

def format_size(size_in_bytes):
    """
    Converts a file size in bytes to a human-readable string format.

    Args:
        size_in_bytes (int): The size of the file in bytes.

    Returns:
        str: The formatted size string, including the appropriate unit (B, KB, MB, GB, TB).
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024

def scale_down_canvas(canvas, scale_factor):
    """
    Scales down an image canvas by a specified factor using interpolation.

    Args:
        canvas (numpy.ndarray): The original image canvas to be scaled down.
        scale_factor (int): The factor by which to scale down the image. 
                            A value of 1 means no scaling (returns the original image).

    Returns:
        numpy.ndarray: The scaled-down image canvas.
    """
    if scale_factor == 1:
        return canvas
    else:
        downsampled_canvas = cv2.resize(
            canvas,
            (canvas.shape[1] // scale_factor, canvas.shape[0] // scale_factor),
            interpolation=cv2.INTER_AREA
        )
        return downsampled_canvas

def save_image(image, filename, scale_down_factor=1):
    """
    Saves an image to a file after optionally scaling it down.

    Args:
        image (numpy.ndarray): The image to be saved.
        filename (str): The name of the file to save the image to.
        scale_down_factor (int, optional): The factor by which to scale down the image before saving. 
                                           Defaults to 1 (no scaling).

    Returns:
        None
    """
    file, ending = filename.split(".")
    filename = file + datetime.now().strftime("_%Y-%m-%d_%H%M%S") + "." + ending
    cv2.imwrite(filename, scale_down_canvas(image, scale_down_factor))
    print(format_size(os.path.getsize(filename)))


def stage_setup(home=True):
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
        raise ConnectionError("Failed to connect to the stage controller.")

    # Ensure the device is successfully opened
    if mcm301obj.is_open(sn) == 0:
        print("MCM301IsOpen failed")
        mcm301obj.close()
        exit()

    # Home the stages
    if home:
        for stage_num in (4,5):
            print(f"Homing stage {stage_num}")
            mcm301obj.home(stage_num)

        # Wait for homing to complete by checking the status bits
        bits_x, bits_y = [0], [0]
        while bits_x[0] not in config.CONFIRMATION_BITS or bits_y[0] not in config.CONFIRMATION_BITS:
            mcm301obj.get_mot_status(4, [0], bits_x)
            mcm301obj.get_mot_status(5, [0], bits_y)
            # print(f"x: {bits_x}, y:{bits_y}")
    
        print("Homing complete")
        print("Stage setup complete\n")

        move(mcm301obj, (1e6, 1e6), wait=False)

    return mcm301obj

def rgb2hex(color):
    r, g, b = color
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def move(mcm301obj, pos, stages=(4, 5), wait=True):
    """
    Moves the stage to a specified position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (tuple): The desired position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
        wait (bool): Whether to wait for the movement to complete before returning.
   
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

    if wait:
        moving = True
        while moving:
            moving = False
            for i, stage in enumerate(stages):
                # Wait until the stages have finished moving by checking the status bits
                bit = [0]
                mcm301obj.get_mot_status(stage, [0], bit)
                if bit[0] not in config.CONFIRMATION_BITS:
                    moving = True
                # print(bit[0])



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


def move_relative(mcm301obj, pos=[0, 0], stages=(4, 5), wait=True):
    """
    Moves the stage to a specified position relative to the current position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (list): The desired relative position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers between 4 and 6 (e.g., 4 for X-axis and 5 for Y-axis).
        wait (bool): Whether to wait for the movement to complete before returning.
   
    The function retrieves the current position of the specified stage, adds the relative position to it,
    and then moves the stage to the new position. It continues to check the status of the stage
    until it confirms that the movement is complete.
    """
    pos = [p + c for p, c in zip(pos, get_pos(mcm301obj, stages))] 
    move(mcm301obj, pos, stages, wait)



def get_scan_area(mcm301obj):
    """
    Retrieves the start and end positions for the scan algorithm.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
   
    Returns:
        start, end: Lists of positions corresponding to the start and end position,
                    in nanometers.
   
    The function queries the current encoder value for each specified stage,
    converts that value into nanometers, and returns the positions as a list.
    """
    input("Please move the stage to one corner of the sample. Press ENTER when complete")
    x_1, y_1 = get_pos(mcm301obj, (4, 5))
    input("Please move the stage to the opposite corner of the sample. Press ENTER when complete")
    x_2, y_2 = get_pos(mcm301obj, (4, 5))
    start = [min(x_1, x_2), min(y_1, y_2)]
    end = [max(x_1, x_2), max(y_1, y_2)]
    print()
    return start, end


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
            aspect = image.size[0]/image.size[1]

            # resize image
            image = image.resize((int(500*aspect),500))
            
            
            self._image = ImageTk.PhotoImage(master=self, image=image)
            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                # resize the canvas to match the new image size
                self._image_width = image.size[0] #self._image.width()
                self._image_height = image.size[1] #self._image.height()
                self.config(width=self._image_width, height=self._image_height)
            self.create_image(0, 0, image=self._image, anchor='nw')
        except queue.Empty:
            pass
        self.after(10, self._get_image)

class ImageAcquisitionThread(threading.Thread):
    def __init__(self, camera, rotation_angle=0):
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._previous_timestamp = 0
        self._rotation_angle = rotation_angle  # New parameter for rotation

        logging.debug("Initializing ImageAcquisitionThread...")

        # setup color processing if necessary
        if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            # Sensor type is not compatible with the color processing library
            self._is_color = False
            logging.debug("Camera is monochrome.")
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
            logging.debug("Camera is color. MonoToColorProcessor initialized.")

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0  # Do not want to block for long periods of time
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        logging.debug("ImageAcquisitionThread initialized.")

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
        color_image_data = self._mono_to_color_processor.transform_to_24(
            frame.image_buffer,
            self._image_width,
            self._image_height
        )
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)
        
        # Create a PIL Image object
        pil_image = Image.fromarray(color_image_data, mode='RGB')


        # Rotate the image by the specified angle
        if self._rotation_angle != 0:
            pil_image = pil_image.rotate(self._rotation_angle, expand=True)


        return pil_image


    def _get_image(self, frame):
        # type: (Frame) -> Image
        # no coloring, just scale down image to 8 bpp and place into PIL Image object
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        pil_image = Image.fromarray(scaled_image)


        # Rotate the image by the specified angle
        if self._rotation_angle != 0:
            pil_image = pil_image.rotate(self._rotation_angle, expand=True)


        return pil_image

    def run(self):
            logging.info("Image acquisition thread started.")
            while not self._stop_event.is_set():
                try:
                    self._camera.issue_software_trigger()
                    frame = self._camera.get_pending_frame_or_null()
                    if frame is not None:
                        # logging.debug("Frame received.")
                        if self._is_color:
                            pil_image = self._get_color_image(frame)
                        else:
                            pil_image = self._get_image(frame)
                        self._image_queue.put_nowait(pil_image)
                    else:
                        # logging.debug("No frame received. Retrying...")
                        time.sleep(0.01)
                except TLCameraError as error:
                    logging.exception("Camera error encountered in image acquisition thread:")
                    break
                except queue.Full:
                    # logging.warning("Image queue is full. Dropping frame.")
                    pass
                except Exception as error:
                    logging.exception("Encountered error in image acquisition thread:")
                    break
            logging.info("Image acquisition thread has stopped.")
            if self._is_color:
                self._mono_to_color_processor.dispose()
                self._mono_to_color_sdk.dispose()
