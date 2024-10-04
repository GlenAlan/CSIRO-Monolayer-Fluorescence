import os
import sys
import signal
import time
import random
import math
import typing
import threading
import queue
import operator
import logging
import platform
import warnings
from datetime import datetime, timedelta
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
import torch

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser, font as tkFont

# Image processing and visualization
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageColor
import cv2 
import numpy as np
from skimage.measure import shannon_entropy

# Thorlabs TSI SDK imports
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame, TLCameraError
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

# Mathematical and scientific tools
from scipy.fftpack import fft2, fftshift
from scipy.optimize import curve_fit
from scipy.signal.windows import hamming

# Custom modules
import config


from MCM301_COMMAND_LIB import *

# Windows-specific setup (optional)
try:
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

# System detection
OS = platform.system()


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all types of logs
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app_debug.log', mode='w')  # Log to a file
    ]
)

# Define image PIL image rotation based on the view number config
view_num_to_rotation = {
                0: None,
                1: Image.ROTATE_90,
                2: Image.ROTATE_180,
                3: Image.ROTATE_270
            }

# Define image cv2 image rotation based on the view number config
view_num_to_rotation_cv = {
                0: None,
                1: cv2.ROTATE_90_COUNTERCLOCKWISE,
                2: cv2.ROTATE_180,
                3: cv2.ROTATE_90_CLOCKWISE
            }


def image_to_stage(center_coords, start, factor=1):
    """
    Converts image coordinates to stage coordinates based on a scaling factor and camera dimensions.
    
    Args:
        center_coords (tuple): The center coordinates of the object in image space (e.g., in pixels).
        start (tuple): The starting position of the stage in nanometers.
        factor (int, optional): A scaling factor to adjust the image coordinates, default is 1.
    
    Returns:
        tuple: The calculated raw stage coordinates in nanometers.
    
    The function takes the image coordinates and adjusts them using a scaling factor. It then 
    computes the corresponding stage coordinates by considering the camera's dimensions and the 
    number of nanometers per pixel (NM_PER_PX). The result is offset by the starting stage position 
    to produce the final stage coordinates in nanometers.
    """
    
    # Scale the center coordinates using the given factor
    x, y = factor * center_coords[0], factor * center_coords[1] 
    
    # Calculate the raw stage coordinates in nanometers by adjusting the center coordinates 
    # based on camera dimensions, pixel-to-nanometer ratio, and the start position.
    center_coords_raw = int((x - config.CAMERA_DIMS[0]) * config.NM_PER_PX + start[0]), \
                        int((y - config.CAMERA_DIMS[1]) * config.NM_PER_PX + start[1])
    
    # Return the calculated raw stage coordinates
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
    Initializes and sets up the stage for movement by creating an instance of the MCM301 object
    and connecting to the first available device. If the device is successfully opened, 
    the function optionally homes the stages (4 and 5), moving them to their reference positions.
    
    Args:
        home (bool, optional): Whether to home the stages after initialization. Defaults to True.
    
    Returns:
        mcm301obj (MCM301): The initialized MCM301 object ready for further operations.
    
    The function performs the following steps:
    1. Initializes the stage controller object.
    2. Lists connected devices and attempts to connect to the first one found.
    3. Verifies that the device is open; if it isn't, the script exits.
    4. If `home` is True, it homes stages 4 and 5, waits for the homing process to complete,
       and moves the stages to an initial position.
    """
    
    # Create an instance of the MCM301 class to control the stage.
    mcm301obj = MCM301()

    # List all connected MCM301 devices.
    devs = MCM301.list_devices()
    print(devs)
    
    # Exit if no devices are found.
    if len(devs) <= 0:
        print('There is no devices connected')
        exit()

    # Connect to the first available device by opening the connection.
    device_info = devs[0]
    sn = device_info[0]  # Extract the serial number of the first device.
    print("connect ", sn)
    
    # Open the device connection with the specified serial number, baud rate (115200), and timeout.
    hdl = mcm301obj.open(sn, 115200, 3)
    if hdl < 0:
        raise ConnectionError("Failed to connect to the stage controller.")

    # Check if the device has been successfully opened.
    if mcm301obj.is_open(sn) == 0:
        print("MCM301IsOpen failed")
        mcm301obj.close()  # Close the connection if unsuccessful.
        exit()

    # If homing is required, proceed with homing the stages.
    if home:
        # Home stages 4 and 5
        for stage_num in (4, 5):
            print(f"Homing stage {stage_num}")
            mcm301obj.home(stage_num)

        # Wait until both stages have completed homing.
        bits_x, bits_y = [0], [0]
        while bits_x[0] not in config.CONFIRMATION_BITS or bits_y[0] not in config.CONFIRMATION_BITS:
            # Continuously check the status of each stage until the confirmation bits indicate completion.
            mcm301obj.get_mot_status(4, [0], bits_x)
            mcm301obj.get_mot_status(5, [0], bits_y)
            # print(f"x: {bits_x}, y:{bits_y}") 

        print("Homing complete")
        print("Stage setup complete\n")

        # After homing, move the stages to the position (1e6, 1e6) nanometers
        move(mcm301obj, (1e6, 1e6), wait=False)

    return mcm301obj


def rgb2hex(color):
    """
    Converts an RGB color value to its hexadecimal string representation.
    
    Args:
        color (tuple): A tuple representing the RGB color, where each value (r, g, b) is an integer between 0 and 255.
    
    Returns:
        str: A string representing the color in hexadecimal format, prefixed with '#'.
    
    The function extracts the red (r), green (g), and blue (b) values from the input tuple, then formats them
    into a hexadecimal string using the `{:02x}` format, ensuring that each color component is represented
    by exactly two hexadecimal digits.
    """
    
    # Unpack the RGB color tuple into separate red (r), green (g), and blue (b) values.
    r, g, b = color
    
    # Format the RGB values as a hexadecimal string with two digits for each color component.
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def convert_encoder_to_nm_with_retries(mcm301obj, slot, encoder_count, nm, retries=10, delay=0.01):
    """ Convert raw encoder to nm with retries.

    Args:
        slot (int): Target slot (4,5,6).
        encoder_count (int): Encoder count.
        nm (list): List to store the converted nm value.
        retries (int): Number of retry attempts.
        delay (float): Delay between retries in seconds.

    Returns:
        int: 0 on success; negative number on failure.
    """
    for attempt in range(retries):
        ret = mcm301obj.convert_encoder_to_nm(slot, encoder_count, nm)
        if ret == 0:
            return 0  # Success
        else:
            logging.warning(f"convert_encoder_to_nm failed on attempt {attempt + 1}/{retries} for slot {slot}. Retrying...")
            time.sleep(delay)
    logging.error(f"convert_encoder_to_nm failed after {retries} attempts for slot {slot}.")
    return ret  # Return the last error code


def convert_nm_to_encoder_with_retries(mcm301obj, slot, nm_position, encoder_count, retries=10, delay=0.01):
    """ Convert nm to raw encoder count with retries.

    Args:
        slot (int): Target slot (4,5,6).
        nm_position (float): Position in nanometers.
        encoder_count (list): List to store the encoder count.
        retries (int): Number of retry attempts.
        delay (float): Delay between retries in seconds.

    Returns:
        int: 0 on success; negative number on failure.
    """
    for attempt in range(retries):
        ret = mcm301obj.convert_nm_to_encoder(slot, nm_position, encoder_count)
        if ret == 0:
            return 0  # Success
        else:
            logging.warning(f"convert_nm_to_encoder failed on attempt {attempt + 1}/{retries} for slot {slot}. Retrying...")
            time.sleep(delay)
    logging.error(f"convert_nm_to_encoder failed after {retries} attempts for slot {slot}.")
    return ret  # Return the last error code




def get_pos(mcm301obj, stages=(4, 5, 6), retries=10, delay=0.01):
    """
    Retrieves the current position of the specified stages with retries.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        stages (tuple): A tuple of stage numbers for which the positions are to be retrieved.
        retries (int): Number of retry attempts for each stage.
        delay (float): Delay between retries in seconds.

    Returns:
        pos (list): A list of positions corresponding to the specified stages, in nanometers.
    """
    pos = []
    for stage in stages:
        encoder_val = [0]
        nm = [0]

        for attempt in range(retries):
            # Get the current encoder value for the stage
            ret_status = mcm301obj.get_mot_status(stage, encoder_val, [0])
            if ret_status != 0:
                logging.warning(f"Failed to get motor status for stage {stage} on attempt {attempt + 1}/{retries}. Retrying...")
                time.sleep(delay)
                continue

            # Convert the encoder value to nanometers with retries
            ret_convert = convert_encoder_to_nm_with_retries(mcm301obj, stage, encoder_val[0], nm)

            # Successful retrieval and conversion
            pos.append(nm[0])
            break  # Exit the retry loop for this stage

        else:
            # If all retries failed
            logging.error(f"Failed to get position for stage {stage} after {retries} attempts.")
            pos.append(None)  # Append None to indicate failure

    return pos



def move(mcm301obj, pos, stages=(4, 5), wait=True):
    """
    Moves the stage to a specified position and waits for the movement to complete.

    Args:
        mcm301obj (MCM301): The MCM301 object that controls the stage.
        pos (tuple): The desired position to move to, given as a tuple of coordinates in nanometers.
        stages (tuple): The stages to move, represented by integers (e.g., 4 for X-axis and 5 for Y-axis).
        wait (bool): Whether to wait for the movement to complete before returning.
    """
    print(f"Moving to {', '.join(str(p) for p in pos)}")

    if len(pos) != len(stages):
        print("Error: Position and stages length mismatch")
        return

    for i, stage in enumerate(stages):
        coord = [0]
        position_nm = float(pos[i])

        # Convert the positions from nanometers to encoder units with retries
        ret = convert_nm_to_encoder_with_retries(mcm301obj, stage, position_nm, coord)
        if ret != 0:
            logging.error(f"Failed to convert {position_nm} nm to encoder units for stage {stage}. Retrying...")
            continue  # Skip moving this stage

        encoder_units = coord[0]
        if encoder_units < 0 or encoder_units > 2147483647:
            logging.warning(f"Encoder units {encoder_units} out of bounds for stage {stage}")
            continue  # Skip moving this stage

        print(f"Stage {stage}: {position_nm} nm -> {encoder_units} encoder units")

        # Move the stage to the required encoder position
        ret_move = mcm301obj.move_absolute(stage, encoder_units)
        if ret_move != 0:
            logging.error(f"Failed to move stage {stage} to {encoder_units} encoder units. Error code: {ret_move}")
            continue

    if wait:
        moving = True
        while moving:
            moving = False
            for i, stage in enumerate(stages):
                # Wait until the stages have finished moving by checking the status bits
                encoder_val = [0]
                status_bits = [0]
                ret_status = mcm301obj.get_mot_status(stage, encoder_val, status_bits)

                if status_bits[0] not in config.CONFIRMATION_BITS:
                    moving = True


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


@njit(parallel=True)
def bin_image_numba(image_array: np.ndarray, binx: int, biny: int) -> np.ndarray:
    """
    Reduces the resolution of an image and increased the exposure and noise by binning pixels along both axes.
    
    This function bins an image by summing groups of adjacent pixels into a single pixel. The image is processed 
    in parallel using Numba's JIT compilation and parallel execution for faster performance.

    Args:
        image_array (np.ndarray): The input image array with shape (height, width, channels).
        binx (int): The binning factor along the x-axis (width).
        biny (int): The binning factor along the y-axis (height).

    Returns:
        np.ndarray: The binned image, with reduced resolution, and the same number of channels.
    
    Raises:
        ValueError: If binx or biny is less than 1.
    
    The function works by dividing the input image into non-overlapping bins of size (binx, biny). It calculates 
    the total pixel value for each bin and creates a new image with reduced dimensions and increased exposure.
    """

    # Validate binning factors
    if binx < 1 or biny < 1:
        raise ValueError("Binning factors binx and biny must be positive integers.")

    # Get the dimensions of the input image
    height, width, channels = image_array.shape

    # Calculate the dimensions of the binned image
    new_height = height // biny
    new_width = width // binx

    # Initialize the output binned image with zeros
    binned_image = np.zeros((new_height, new_width, channels), dtype=np.float32)

    # Loop through each pixel in the new binned image
    for y in prange(new_height):  # Use prange for parallel execution across rows (height)
        for x in range(new_width): 
            for c in range(channels):  # Loop over color channels
                sum_val = 0.0 

                # Sum pixel values from the corresponding bin in the original image
                for dy in range(biny): 
                    for dx in range(binx):
                        sum_val += image_array[y * biny + dy, x * binx + dx, c]

                # Assign the pixel sum to the new binned image
                binned_image[y, x, c] = sum_val

    # Clip the values to ensure they are within the valid range [0, 255] and convert to uint8 type
    binned_image = np.clip(binned_image, 0, 255).astype('uint8')

    return binned_image


def bin_image_cv2(image_array: np.ndarray, binx: int, biny: int) -> np.ndarray:
    """
    Bins an image using OpenCV to reduce its resolution.

    Args:
        image_array (np.ndarray): The input image with shape (height, width, channels).
        binx (int): The binning factor along the x-axis (width).
        biny (int): The binning factor along the y-axis (height).

    Returns:
        np.ndarray: The binned image with reduced resolution.
    
    Raises:
        ValueError: If binx or biny is less than 1.
    
    This method reduces the image size using OpenCV's INTER_AREA interpolation, 
    which is well-suited for downscaling.
    """
    
    # Validate binning factors
    if binx < 1 or biny < 1:
        raise ValueError("Binning factors binx and biny must be positive integers.")
    
    # Get original image dimensions
    height, width, channels = image_array.shape

    # Calculate new dimensions after binning
    new_width = width // binx
    new_height = height // biny
    
    # Resize the image using OpenCV's INTER_AREA interpolation
    binned_image = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return binned_image



def bin_image(image_array: np.ndarray, binx: int, biny: int) -> np.ndarray:
    """
    Combines two binning methods: Numba's binning and OpenCV's resizing.
    This is done to balance the exposure increase without adding too much noise.

    Args:
        image_array (np.ndarray): The input image with shape (height, width, channels).
        binx (int): The binning factor along the x-axis (width).
        biny (int): The binning factor along the y-axis (height).

    Returns:
        np.ndarray: The image after being binned by both Numba and OpenCV.
    
    First, it bins the image using a custom Numba function, then applies further
    resizing using OpenCV for efficient downsampling.
    """
    
    # First bin using Numba for better exposure, then apply additional binning with OpenCV
    if binx == biny == 1:
        return image_array
    elif config.BINNING_EXPOSURE_INCREASE:
        return bin_image_cv2(bin_image_numba(image_array, binx, biny), binx, biny)
    else:
        return bin_image_cv2(image_array, binx*2, biny*2)



class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def __init__(self, master=None, orient='vertical', **kwargs):
        # Determine the appropriate style based on orientation
        if orient == 'horizontal':
            style = 'Custom.Horizontal.TScrollbar'
        else:
            style = 'Custom.Vertical.TScrollbar'
        super().__init__(master, style=style, orient=orient, **kwargs)

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            super().set(lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage:
    """ Display and zoom image """
    def __init__(self, placeholder, im, gui):
        """ Initialize the ImageFrame """
        # Initialize styles
        self.style = ttk.Style()
        self.style.theme_use('default')  # Use the default theme as a base
        self.gui = gui

        # Configure custom styles for horizontal and vertical scrollbars
        self.style.configure('Custom.Horizontal.TScrollbar',
                             troughcolor=config.THEME_COLOR,
                             background=config.BUTTON_COLOR,
                             bordercolor=config.THEME_COLOR)
        self.style.configure('Custom.Vertical.TScrollbar',
                             troughcolor=config.THEME_COLOR,
                             background=config.BUTTON_COLOR,
                             bordercolor=config.THEME_COLOR)
        self.style.configure('Custom.TFrame', background=config.THEME_COLOR)

        # Copy the existing layout from the default scrollbar styles
        self.style.layout("Custom.Horizontal.TScrollbar", self.style.layout("Horizontal.TScrollbar"))
        self.style.layout("Custom.Vertical.TScrollbar", self.style.layout("Vertical.TScrollbar"))

        # Initialize image scaling and other parameters
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.LANCZOS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.im = im  # PIL Image object

        # Create ImageFrame in placeholder widget with custom style
        self.__imframe = ttk.Frame(placeholder, style='Custom.TFrame')

        # Configure the placeholder's grid to expand
        placeholder.rowconfigure(0, weight=1)
        placeholder.columnconfigure(0, weight=1)

        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')

        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(
            self.__imframe,
            highlightthickness=0,
            xscrollcommand=hbar.set,
            yscrollcommand=vbar.set,
            width=800,
            height=800,
            bg=config.THEME_COLOR  # Set canvas background to config.THEME_COLOR
        )
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created

        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)

        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<Double-Button-1>', self.__double_click)  # handle double-click for coordinates
        # Handle keystrokes in idle mode, because program slows down on weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))

        # Decide if this image is huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning for big image
            warnings.simplefilter('ignore')
            self.__image = self.im  # open image, but don't load it into RAM

        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        logging.debug(f"Image size: {self.imwidth}x{self.imheight}")
        logging.debug(f"Minimum side: {self.__min_side}")

        # Check if the image is considered "huge"
        if (self.imwidth * self.imheight > self.__huge_size * self.__huge_size and
            hasattr(self.__image, 'tile') and
            self.__image.tile and
            self.__image.tile[0][0] == 'raw'):  # only raw images could be tiled
            self.__huge = True  # image is huge
            logging.debug("Image is huge and tiled as 'raw'.")
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [
                self.__image.tile[0][0],  # it has to be 'raw'
                [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                self.__offset,
                self.__image.tile[0][3]  # list of arguments to the decoder
            ]
        else:
            self.__huge = False
            logging.debug("Image is not huge or does not have 'raw' tiling.")

        # Create image pyramid
        if self.__huge:
            self.__pyramid = [self.smaller()]
        else:
            self.__pyramid = [self.im]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramid scale
        self.__reduction = 2  # reduction degree of image pyramid
        (w, h), m, j = self.__pyramid[-1].size, 512, 0
        n = math.ceil(math.log(min(w, h) / m, self.__reduction)) + 1  # image pyramid length
        while w > m and h > m:  # top pyramid image is around 512 pixels in size
            j += 1
            logging.debug(f"Creating image pyramid: {j} from {n}")
            w /= self.__reduction  # divide by reduction degree
            h /= self.__reduction  # divide by reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        logging.debug("Image pyramid creation complete.")

        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ratio2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 0, math.ceil(self.imheight / self.__band_width)
        while i < self.imheight:
            j += 1
            logging.debug(f"Opening image: {j} from {n}")
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            if self.__huge:
                self.__tile[1][3] = band  # set band width
                self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
                self.__image.close()
                self.__image = self.im  # reopen / reset image
                self.__image.size = (self.imwidth, band)  # set size of the tile band
                self.__image.tile = [self.__tile]  # set tile
                cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
                image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
        logging.debug("Image resizing complete.")
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (
            self.canvas.canvasx(0),  # get visible area of the canvas
            self.canvas.canvasy(0),
            self.canvas.canvasx(self.canvas.winfo_width()),
            self.canvas.canvasy(self.canvas.winfo_height())
        )
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [
            min(box_img_int[0], box_canvas[0]),
            min(box_img_int[1], box_canvas[1]),
            max(box_img_int[2], box_canvas[2]),
            max(box_img_int[3], box_canvas[3])
        ]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it is in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image, which does not fit in RAM
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = self.im  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (
                        int(x1 / self.__scale),
                        int(y1 / self.__scale),
                        int(x2 / self.__scale),
                        int(y2 / self.__scale)
                    )
                )
            #
            imagetk = ImageTk.PhotoImage(
                image.resize((int(x2 - x1), int(y2 - y1)), self.__filter)
            )
            imageid = self.canvas.create_image(
                max(box_canvas[0], box_img_int[0]),
                max(box_canvas[1], box_img_int[1]),
                anchor='nw',
                image=imagetk
            )
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def __double_click(self, event):
        """ Get the original pixel coordinates of the image on double-click """
        # Get the canvas coordinates (including any panning/scrolling)
        x = self.canvas.canvasx(event.x) - self.canvas.coords(self.container)[0]
        y = self.canvas.canvasy(event.y) - self.canvas.coords(self.container)[1]

        # Scale back to the original image size by dividing by the current zoom (imscale)
        x_original = x / self.imscale
        y_original = y / self.imscale

        # Ensure the coordinates are within the bounds of the original image size
        x_original = max(0, min(self.imwidth, x_original))
        y_original = max(0, min(self.imheight, y_original))

        print(f"Original image coordinates: ({int(x_original)}, {int(y_original)})")
        movement_array = [x_original, y_original, self.imwidth-x_original, self.imheight-y_original]
        final_x, final_y = image_to_stage([int(movement_array[(-config.VIEW_ROTATION)%4]), int(movement_array[(-config.VIEW_ROTATION+1)%4])], config.start_pos, config.RESULTS_IMAGE_DOWNSCALE)
        print(f"Final stage coordinates: ({final_x}, {final_y})")
        move(self.gui.stage_controller, (final_x, final_y), wait=False)

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y):
            return  # zoom only inside image area
        scale = 1.0
        if event.delta or event.num in [4, 5]:  # Windows/MacOS or Linux
            if event.delta < 0 or event.num == 5:  # scroll down, zoom out, smaller
                if round(self.__min_side * self.imscale) < 30:
                    return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale /= self.__delta
            elif event.delta > 0 or event.num == 4:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) / 2)
                if i < self.imscale:
                    return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll', 1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = self.im  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        for img in self.__pyramid:
            img.close()  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()


class ImageAcquisitionThread(threading.Thread):
    """
    A thread that continuously acquires images from the ThorLabs camera and processes them. Images are placed into a queue for further use.

    Args:
        camera (Camera): The camera object from which to acquire images.
        rotation_angle (int, optional): The angle to rotate the captured images, default is 0 degrees. This acts as calibration for the camera orientation.
    """
    
    def __init__(self, camera, rotation_angle=0):
        """
        Initializes the image acquisition thread with the given camera and optional rotation angle.
        """
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._previous_timestamp = 0
        self._rotation_angle = rotation_angle 

        logging.debug("Initializing ImageAcquisitionThread...")

        # Determine if the camera supports color processing
        if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            self._is_color = False
            logging.debug("Camera is monochrome.")
        else:
            # Set up color processing for Bayer sensor
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
        self._camera.image_poll_timeout_ms = 0  # Non-blocking polling for images
        self._image_queue = queue.Queue(maxsize=2)  # Image queue with a limit of 2 images
        self._stop_event = threading.Event()  # Event to stop the thread
        logging.debug("ImageAcquisitionThread initialized.")

    def get_output_queue(self):
        """
        Returns the queue where processed images are stored.
        
        Returns:
            queue.Queue: The image queue.
        """
        return self._image_queue

    def stop(self):
        """
        Signals the thread to stop.
        """
        self._stop_event.set()

    def _get_color_image(self, frame):
        """
        Processes a color image by converting raw Bayer data to RGB, binning, and rotating the image.
        
        Args:
            frame (Frame): The frame containing the raw image data.

        Returns:
            Image: The processed color image as a PIL Image object.
        """
        # Check for changes in image dimensions
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]
        if (width != self._image_width) or (height != self._image_height):
            self._image_width = width
            self._image_height = height
            logging.info("Image dimension change detected, image acquisition thread was updated")

        # Convert monochrome to color (RGB)
        color_image_data = self._mono_to_color_processor.transform_to_24(
            frame.image_buffer,
            self._image_width,
            self._image_height
        )
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)

        # Apply binning
        binx = biny = config.CAMERA_BINNING
        binned_image_data = bin_image(color_image_data, binx, biny)

        # Convert to uint8 format
        binned_image_data = binned_image_data.astype('uint8')

        # Create PIL image
        pil_image = Image.fromarray(binned_image_data, mode='RGB')

        # Rotate the image if required
        if self._rotation_angle != 0:
            pil_image = pil_image.rotate(self._rotation_angle, expand=True)

        return pil_image

    def _get_image(self, frame):
        """
        Processes a monochrome image by scaling it and rotating if needed.
        
        Args:
            frame (Frame): The frame containing the raw image data.

        Returns:
            Image: The processed monochrome image as a PIL Image object.
        """
        # Scale the image to 8 bits per pixel
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        pil_image = Image.fromarray(scaled_image)

        # Rotate the image if required
        if self._rotation_angle != 0:
            pil_image = pil_image.rotate(self._rotation_angle, expand=True)

        return pil_image

    def run(self):
        """
        Main loop of the thread that continuously acquires and processes images until the thread is stopped.
        """
        logging.info("Image acquisition thread started.")
        while not self._stop_event.is_set():
            try:
                # Trigger the camera to capture an image
                self._camera.issue_software_trigger()
                frame = self._camera.get_pending_frame_or_null()  # Get the frame

                if frame is not None:
                    # Process the frame based on whether the camera is color or monochrome
                    if self._is_color:
                        pil_image = self._get_color_image(frame)
                    else:
                        pil_image = self._get_image(frame)

                    # Add the processed image to the queue
                    try:
                        self._image_queue.put_nowait(pil_image)
                    except queue.Full:
                        pass  # Drop the frame if the queue is full
                else:
                    time.sleep(0.01)  # Retry after a short delay if no frame is available
            except TLCameraError as error:
                logging.exception("Camera error encountered in image acquisition thread:")
                break
            except Exception as error:
                logging.exception("Encountered error in image acquisition thread:")
                break
        
        logging.info("Image acquisition thread has stopped.")
        
        # Cleanup color processor resources
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()



class LiveViewCanvas(tk.Canvas):
    """
    A custom Tkinter Canvas that displays live images from a queue, updating the image display based on
    resizing events and controlling image scaling to fit the canvas. Images are updated at a regular interval defined in config.

    Inherits from `tk.Canvas` and adds functionality for live image updating and responsive resizing.
    
    Args:
        parent (tk.Widget): The parent Tkinter widget that this canvas will be placed into.
        image_queue (queue.Queue): A queue holding images to be displayed. The queue is expected to provide PIL images.
    
    Attributes:
        image_queue (queue.Queue): The queue from which images are retrieved.
        is_active (bool): A flag to control whether image updates should continue.
        original_image_width (int): The width of the original image before resizing.
        original_image_height (int): The height of the original image before resizing.
        displayed_image_width (int): The width of the currently displayed image.
        displayed_image_height (int): The height of the currently displayed image.
        current_image (PIL.Image): The current image being displayed.
        displayed_image (PIL.Image): The resized version of the current image being displayed.
        scale (float): The scaling factor used to resize the image.
    """
    
    def __init__(self, parent, image_queue):
        """
        Initializes the LiveViewCanvas with a parent widget and an image queue. It also binds a resizing event
        to the canvas and starts the image update loop.

        Args:
            parent (tk.Widget): The parent widget in which this canvas resides.
            image_queue (queue.Queue): The queue holding the images to be displayed.
        """
        # Initialize the parent canvas, set background color, and remove highlight borders
        super().__init__(parent, bg=config.THEME_COLOR, highlightthickness=0)
        
        self.image_queue = image_queue  # Store the image queue
        self.is_active = True  # Flag to control whether the canvas should continue updating images
        
        # Bind the canvas resize event to the on_resize method
        self.bind("<Configure>", self.on_resize)

        # Start the image display loop
        self._display_image()

    def _display_image(self):
        """
        Continuously retrieves and displays the latest image from the image queue. If no new image is available,
        it retries after a delay defined in the configuration. This method ensures that only the most recent image
        is displayed if multiple images are queued.
        """
        if not self.is_active:
            # If updates are paused, retry after the configured delay
            self.after(config.UPDATE_DELAY, self._display_image)
            return

        image = None  # Initialize image variable
        
        try:
            # Continuously retrieve the latest available image from the queue
            while True:
                try:
                    image = self.image_queue.get_nowait()  # Get an image from the queue without waiting
                except queue.Empty:
                    break  # Exit the loop if no more images are in the queue
        except Exception as e:
            print(f"Error retrieving image from queue: {e}")

        # If an image was successfully retrieved, update the canvas display
        if image:
            self.update_image_display(image)

        # Schedule the next update after a delay
        self.after(config.UPDATE_DELAY, self._display_image)

    def update_image_display(self, image):
        """
        Resizes the given image to fit within the canvas while maintaining the aspect ratio. It also handles
        any necessary rotation based on the current view setting. The image is then drawn on the canvas.

        Args:
            image (PIL.Image): The image to be displayed on the canvas.
        """
        self.current_image = image  # Save the current image for potential redrawing

        # Rotate the image based on the current view setting
        rotation = view_num_to_rotation[config.VIEW_ROTATION % 4]  # Determine the rotation angle based on config
        if rotation is not None:
            image = image.transpose(rotation)  # Apply the rotation if required

        # Check for valid image dimensions
        if image.width <= 0 or image.height <= 0:
            logging.warning(f"Received image with invalid dimensions: {image.width}x{image.height}. Skipping resize.")
            return

        # Store the original image size before resizing
        self.original_image_width = image.width
        self.original_image_height = image.height
        

        # Get the current size of the canvas
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        # Skip if the canvas has zero width or height (e.g., during initial rendering)
        if canvas_width == 0 or canvas_height == 0:
            logging.warning("Canvas width or height is zero. Skipping image display.")
            return

        # Calculate scaling factors to fit the image within the canvas
        scale_x = canvas_width / self.original_image_width
        scale_y = canvas_height / self.original_image_height
        self.scale = min(scale_x, scale_y)  # Use the smaller scaling factor to preserve aspect ratio

        # Ensure scaling factors are positive
        if self.scale <= 0:
            logging.warning(f"Invalid scaling factors: scale_x={scale_x}, scale_y={scale_y}. Skipping resize.")
            return

        # Resize the image using the calculated scaling factor
        new_width = max(int(self.original_image_width * self.scale), 1)  # Ensure minimum width is 1 pixel
        new_height = max(int(self.original_image_height * self.scale), 1)  # Ensure minimum height is 1 pixel
        self.displayed_image = image.resize((new_width, new_height), Image.LANCZOS)  # Resize the image with high-quality resampling

        # Store the displayed image size for reference
        self.displayed_image_width = new_width
        self.displayed_image_height = new_height

        # Clear the canvas before drawing the new image
        self.delete("all")

        # Convert the resized image into a format compatible with Tkinter
        self.photo_image = ImageTk.PhotoImage(self.displayed_image)

        # Center the image on the canvas
        self.image_x0 = (canvas_width - new_width) // 2  # X-coordinate for centering
        self.image_y0 = (canvas_height - new_height) // 2  # Y-coordinate for centering
        self.create_image(self.image_x0, self.image_y0, anchor='nw', image=self.photo_image)  # Draw the image

    def on_resize(self, event):
        """
        Redraws the current image when the canvas is resized. This ensures that the image is always
        properly scaled to fit the new canvas size.
        
        Args:
            event (tk.Event): The resize event triggered when the canvas is resized.
        """
        # Redraw the image using the updated canvas dimensions if an image is currently being displayed
        if hasattr(self, 'current_image'):
            self.update_image_display(self.current_image)


def add_image_to_canvas(canvas, image, center_coords, alpha=0.5):
    """
    Adds an image with an alpha channel to the canvas at the specified center coordinates using PyTorch for GPU acceleration.
    The image is placed with full opacity except in overlapping regions, where it is blended with transparency.

    Args:
        canvas (numpy.ndarray): The canvas where the image will be overlaid, including an alpha channel.
        image (numpy.ndarray): The image to be added, including an alpha channel.
        center_coords (tuple): The (x, y) coordinates of the center of the image on the canvas.
        alpha (float): The blending factor for overlapping regions, ranging from 0 (transparent) to 1 (opaque).

    Returns:
        numpy.ndarray: The updated canvas with the image blended in.
    """
    # Ensure that the image has an alpha channel
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    # Determine the smaller dimension and crop the image to a square
    min_side = min(image.shape[0], image.shape[1])
    cropped_image = image[:min_side, :min_side]

    img_height, img_width = cropped_image.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]

    # Calculate start and end coordinates for the overlay
    x_center, y_center = center_coords
    x_start = max(0, x_center - img_width // 2)
    y_start = max(0, y_center - img_height // 2)
    x_end = min(canvas_width, x_start + img_width)
    y_end = min(canvas_height, y_start + img_height)

    # Calculate the region of the canvas and image that will be added
    x_offset = x_start - (x_center - img_width // 2)
    y_offset = y_start - (y_center - img_height // 2)
    region_width = x_end - x_start
    region_height = y_end - y_start

    # Convert the relevant regions to PyTorch tensors for CPU/GPU processing
    canvas_region = torch.tensor(canvas[y_start:y_end, x_start:x_end], dtype=torch.float32, device=config.ACCEL_DEVICE)
    image_region = torch.tensor(cropped_image[y_offset:y_offset + region_height, x_offset:x_offset + region_width], dtype=torch.float32, device=config.ACCEL_DEVICE)

    # Separate the alpha channels and normalize them
    image_alpha = (image_region[:, :, 3] / 255.0) * alpha
    canvas_alpha = canvas_region[:, :, 3] / 255.0

    # Compute the new alpha channel after blending
    new_alpha = image_alpha + canvas_alpha * (1 - image_alpha)
    new_alpha_safe = torch.clamp(new_alpha, min=1e-6)

    # Determine where the canvas is initially transparent
    canvas_transparent_mask = (canvas_alpha == 0)

    # Directly place the image where the canvas is transparent
    canvas_region[canvas_transparent_mask] = image_region[canvas_transparent_mask]

    # Blend the regions where there is overlap (both canvas and image have alpha)
    blend_mask = (canvas_alpha > 0) & (image_alpha > 0)

    # Blend RGB channels where there is overlap
    for c in range(3):
        canvas_region[:, :, c][blend_mask] = (
            image_region[:, :, c][blend_mask] * image_alpha[blend_mask] +
            canvas_region[:, :, c][blend_mask] * canvas_alpha[blend_mask] * (1 - image_alpha[blend_mask])
        ) / new_alpha_safe[blend_mask]

    # Update alpha to full opacity after blending (255) for areas that have been blended
    canvas_region[:, :, 3][blend_mask] = new_alpha[blend_mask] * 255

    # Convert the blended result back to a NumPy array
    canvas[y_start:y_end, x_start:x_end] = canvas_region.cpu().numpy().astype(canvas.dtype)

    return canvas


def process_image(item, canvas, start, lock):
    """
    Processes a single image from the queue, adds it to the canvas.

    Args:
        item (tuple): A tuple containing the image and its center coordinates.
        canvas (numpy.ndarray): The canvas where the image will be overlaid, including an alpha channel.
        start (tuple): The (x, y) starting coordinates of the scan area in nanometers.
        lock (threading.Lock): A lock to ensure thread-safe updates to the canvas.

    Returns:
        None: Updates the shared canvas in-place.
    """

    image, center_coords_raw = item

    # Convert the image to a numpy array and ensure it has an alpha channel
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)

    # Calculate the coordinates on the canvas where the image center should be placed
    center_coords = (
        int((center_coords_raw[0] - start[0]) / config.NM_PER_PX + config.CAMERA_DIMS[0]),  # X coordinate
        int((center_coords_raw[1] - start[1]) / config.NM_PER_PX + config.CAMERA_DIMS[1])   # Y coordinate
    )

    # Add the image to the canvas within a lock to ensure thread-safe operation
    with lock:
        add_image_to_canvas(canvas, image_np, center_coords)
        

def stitch_and_display_images(gui, frame_queue, start, end, stitched_view_canvas):
    """
    Continuously stitches and displays images from a queue, blending them into a larger canvas
    that includes an alpha channel for transparency. Saves the final stitched image when all frames are processed.
    
    Args:
        frame_queue (queue.Queue): A queue containing tuples of images and their corresponding center coordinates.
        start (tuple): The (x, y) starting coordinates of the scan area in nanometers.
        end (tuple): The (x, y) ending coordinates of the scan area in nanometers.
    """

    # Assuming square frames
    frame_dims = min(config.CAMERA_DIMS)  
    
    # Calculate the size of the output canvas based on the scan area and the camera dimensions
    output_size = [
        int((end[0] - start[0]) / config.NM_PER_PX + frame_dims * 2.5),  # Width of the canvas
        int((end[1] - start[1]) / config.NM_PER_PX + frame_dims * 2.5)   # Height of the canvas
    ]
    # Initialize the canvas
    canvas = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)  # 4 channels (RGBA)

    # Lock for thread-safe updates to the canvas
    lock = threading.Lock()

    # Using ThreadPoolExecutor to parallelize processing of images
    with ThreadPoolExecutor() as executor:
        futures = []

        while True:
            # Get the next item from the frame queue
            item = frame_queue.get()
            
            if item is None:
                # Signal received that all frames are processed
                break

            # Submit a new task to the executor
            future = executor.submit(process_image, item, canvas, start, lock)
            futures.append(future)

            scaled_canvas = scale_down_canvas(canvas, 8)

            stitched_image = Image.fromarray(cv2.cvtColor(scaled_canvas, cv2.COLOR_BGRA2RGBA))
            stitched_view_canvas.update_image_display(stitched_image)

        # Wait for all futures to complete
        for future in futures:
            future.result()

    print("Saving final image...")

    save_image(canvas, "Images/final_scan.png", 2)

    print("Save complete")

    scaled_canvas = scale_down_canvas(canvas, 4)
    stitched_image = Image.fromarray(cv2.cvtColor(scaled_canvas, cv2.COLOR_BGRA2RGBA))
    stitched_view_canvas.update_image_display(stitched_image)

    config.canvas = canvas

    if config.SAVE_RAW_IMAGE:
        threading.Thread(target=lambda: save_image(canvas, "Images/RAW.png"), daemon=True).start()

    # Pass the final stitched image to the post processing function
    post_processing(gui, canvas)


def post_processing(gui, canvas):
    """
    Handles post-processing of the image on the canvas. This involves scaling the image, applying
    filters, detecting monolayers, and highlighting them on the canvas. Results are displayed and saved.

    Args:
        gui: The GUI object to update the interface and display results.
        canvas: The image canvas to process and extract monolayers from.
    """
    t1 = time.time()  # Start timer for performance tracking
    gui.root.after(0, lambda: gui.update_progress(95, "Image processing..."))
    print("Post processing...")

    monolayers = []

    # Load post-processing configurations
    contrast = config.POST_PROCESSING_CONTRAST
    threshold = config.POST_PROCESSING_THRESHOLD
    blur = config.POST_PROCESSING_BLUR

    # Downscale the image for faster post-processing
    post_image = scale_down_canvas(canvas, config.POST_PROCESSING_DOWNSCALE)

    # Convert the image to greyscale with a bias for red, green, and blue based on configuration
    red, green, blue = [(c / 127) - 1 for c in config.MONOLAYER_COLOR]
    print(f"{red}, {green}, {blue}")

    # Create a weighted grayscale image
    post_image = red * post_image[:, :, 2] + green * post_image[:, :, 1] + blue * post_image[:, :, 0]

    # Normalize the grayscale image to 0-255 range
    post_image = np.clip(post_image, 0, 255).astype(np.uint8)

    # Apply a Gaussian blur to reduce noise
    post_image = cv2.blur(post_image, (blur, blur))

    # Increase image contrast
    post_image = cv2.convertScaleAbs(post_image, alpha=contrast, beta=0)

    # Apply a threshold to remove pixel values below the threshold
    _, post_image = cv2.threshold(post_image, threshold, 255, cv2.THRESH_BINARY)

    t2 = time.time()  # End timer
    print(f"Time taken: {t2 - t1:.2f} seconds")

    # Create a copy of the original canvas for drawing contours
    contour_image = canvas.copy()

    # Update progress and detect contours (monolayers)
    print("Locating Monolayers...")
    gui.root.after(0, lambda: gui.update_progress(97, "Locating Monolayers..."))
    scaled_contours, _ = cv2.findContours(post_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale contours back to the original image size
    contours = [contour * config.POST_PROCESSING_DOWNSCALE for contour in scaled_contours]

    larger_contours = []

    for i, contour in enumerate(contours):
        # Get the bounding box around each monolayer
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding to the bounding box and crop the monolayer from the canvas
        x_start = max(x - config.MONOLAYER_CROP_PADDING, 0)
        y_start = max(y - config.MONOLAYER_CROP_PADDING, 0)
        x_end = min(x + w + config.MONOLAYER_CROP_PADDING, canvas.shape[1])
        y_end = min(y + h + config.MONOLAYER_CROP_PADDING, canvas.shape[0])

        # Crop the monolayer image
        image_section = canvas[y_start:y_end, x_start:x_end]

        # Rotate the cropped section if required
        rotation = view_num_to_rotation_cv[config.VIEW_ROTATION % 4]
        if rotation is not None:
            image_section = cv2.rotate(image_section, rotation)

        # Create a Monolayer object and store it
        monolayers.append(Monolayer(contour, image_section, (x_start, y_start)))

        # Draw a marker at the center of the monolayer
        cx, cy = monolayers[-1].position
        contour_image = cv2.circle(contour_image, (cx, cy), config.MONOLAYER_DOT_SIZE, color=(0, 0, 0, 255), thickness=-1)

        # Scale up the contours if necessary and draw them on the canvas
        if config.MONOLAYER_OUTLINE_SCALE != 1:
            larger_contour = []
            for point in contour:
                x, y = point[0]
                dx = x - cx
                dy = y - cy
                new_x = cx + dx * config.MONOLAYER_OUTLINE_SCALE
                new_y = cy + dy * config.MONOLAYER_OUTLINE_SCALE
                larger_contour.append([int(new_x), int(new_y)])
            larger_contour = np.array(larger_contour, dtype=np.int32).reshape(-1, 1, 2)
            larger_contours.append(larger_contour)
            cv2.drawContours(contour_image, larger_contours, -1, (255, 255, 0, 255), config.MONOLAYER_OUTLINE_THICKNESS)
        else:
            cv2.drawContours(contour_image, contours, -1, (255, 255, 0, 255), config.MONOLAYER_OUTLINE_THICKNESS)

    # Save and display the final processed image with highlighted monolayers
    gui.root.after(0, lambda: gui.update_progress(97, "Saving final images..."))
    print("Saving image with monolayers...")
    save_image(contour_image, "Images/highlighted_monolayers.png", 1)
    print("Saved!")

    # Filter out small monolayers and sort the remaining by area
    monolayers = [layer for layer in monolayers if layer.area_um >= config.POST_PROCESSING_DOWNSCALE]
    monolayers.sort(key=operator.attrgetter('area'), reverse=True)

    # Display the results in the GUI
    gui.root.after(0, lambda: gui.display_results_tab(
        Image.fromarray(cv2.cvtColor(scale_down_canvas(contour_image, config.RESULTS_IMAGE_DOWNSCALE), cv2.COLOR_BGRA2RGBA)), monolayers))

    # Print detailed information about each monolayer
    for i, layer in enumerate(monolayers):
        print(f"{i+1}: Area: {layer.area_um:.0f} um^2,  Centre: {layer.position}      Entropy: {layer.smoothed_entropy:.2f}, TV Norm: {layer.total_variation_norm:.2f}, Local Intensity Variance: {layer.local_intensity_variance:.2f}, CNR: {layer.contrast_to_noise_ratio:.2f}, Skewness: {layer.skewness:.2f}")
        # cv2.imwrite(f"Images/Monolayers/{i+1}.png", layer.image)

    # Final progress update
    gui.root.after(0, lambda: gui.update_progress(100, "Scan Complete!"))



def alg(gui, mcm301obj, image_queue, frame_queue, start, end):
    """
    Runs the main scanning algorithm, capturing images across a defined area for stitching.
    
    Args:
        mcm301obj (MCM301): The stage controller object for controlling movement.
        image_queue (queue.Queue): Queue containing images captured by the camera.
        frame_queue (queue.Queue): Queue to store images along with their positions for stitching.
        start (tuple): Starting coordinates (x, y) in nanometers.
        end (tuple): Ending coordinates (x, y) in nanometers.
    """
    
    # Calculate the total number of images to capture based on the scanning area and step size
    num_images = math.ceil((end[0] - start[0]) / config.DIST + 1) * math.ceil((end[1] - start[1]) / config.DIST + 1)
    print(f"Total images to capture: {num_images}")

    config.current_image = 0 
    focuses = [] # List to store focus values for autofocus

    def capture_and_store_frame(x, y, focuses):
        """
        Captures a frame from the image queue and stores it in the frame queue with its position.
        
        Args:
            x (int): The x-coordinate in nanometers.
            y (int): The y-coordinate in nanometers.
            focuses (list): A list to track focus values.
        """
        # Simulate delay based on camera exposure time
        time.sleep(0.5 * config.CAMERA_PROPERTIES["exposure"] / 1e6)

        # Retrieve the latest frame from the image queue
        frame = image_queue.get(timeout=1000)

        # Autofocus functionality, if enabled
        if config.AUTOFOCUS:
            # Convert frame to grayscale and calculate intensity contrast
            frame_array = np.array(frame.convert('L'))
            intensity_lower = np.percentile(frame_array, config.AUTOFOCUS_REQUIRED_SUBSTANCE_PERCENT)
            intensity_upper = np.percentile(frame_array, 100 - config.AUTOFOCUS_REQUIRED_SUBSTANCE_PERCENT)
            intensity_range = intensity_upper - intensity_lower

            # Perform focus check based on contrast
            if intensity_range > config.AUTOFOCUS_REQUIRED_CONTRAST:
                focuses.append(get_focus(frame))

                # Trigger autofocus if focus value drops below threshold
                if focuses[-1] < config.FOCUSED_THRESHOLD and len(focuses) > config.FOCUS_FRAME_AVG:
                    auto_focus(mcm301obj, image_queue)
                    focuses.clear()
                    frame = image_queue.get(timeout=1000)
                    focuses.append(get_focus(frame))

                # Check for degrading focus and refocus if needed
                elif len(focuses) > config.FOCUS_FRAME_AVG * 2 and np.average(focuses[0:-config.FOCUS_FRAME_AVG]) > np.average(focuses[-config.FOCUS_FRAME_AVG:-1]) * config.FOCUS_BUFFER:
                    auto_focus(mcm301obj, image_queue)
                    focuses.clear()
                    frame = image_queue.get(timeout=1000)
                    focuses.append(get_focus(frame))

        # Store the captured frame and its position in the frame queue
        frame_queue.put((frame, (x, y)))
        config.current_image += 1 

        # Update progress bar
        gui.root.after(0, lambda: gui.update_progress(int(90 * config.current_image / num_images), f"Capturing Image {config.current_image} of {num_images}"))

    def scan_line(x, y, direction, focuses):
        """
        Scans a single line in the specified direction, capturing images along the way.
        
        Args:
            x (int): The current x-coordinate in nanometers.
            y (int): The current y-coordinate in nanometers.
            direction (int): The direction of the scan (1 for forward, -1 for backward).
            
        Returns:
            int: The updated x-coordinate after completing the line scan.
        """
        # Continue capturing images while moving in the given direction
        while (direction == 1 and x < end[0]) or (direction == -1 and x > start[0]):
            capture_and_store_frame(x, y, focuses)  # Capture an image at the current position
            x += config.DIST * direction 
            move(mcm301obj, (x, y))  # Move the stage to the next point
        
        # Capture the final frame at the end of the line
        capture_and_store_frame(x, y, focuses)
        
        return x

    # Begin scanning process
    move(mcm301obj, start)
    x, y = start 
    direction = 1

    # Disable autofocus button during scanning
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='disabled'))

    # Perform autofocus before starting if enabled
    if config.AUTOFOCUS:
        gui.root.after(0, lambda: gui.update_progress(1, "Focusing..."))
        auto_focus(mcm301obj, image_queue, [config.FOCUS_RANGE * 2, config.FOCUS_STEPS * 2])

    # Scan in lines until the entire area is covered
    while y < end[1]:
        x = scan_line(x, y, direction, focuses) 
        y += config.DIST
        move(mcm301obj, (x, y))
        direction *= -1

    # Perform the final scan to complete the area
    scan_line(x, y, direction, focuses)

    print("\nImage capture complete!")
    print("Waiting for image processing to complete...")

    # Update GUI after image capture and re-enable autofocus button
    gui.root.after(0, lambda: gui.update_progress(92, "Image stitching..."))
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='normal'))

    # Signal the frame queue that scanning is complete
    frame_queue.put(None)


def get_focus(image):
    """
    Calculates the focus measure of an image using the Power Spectrum Slope method.
    This involves taking the 2D Fourier Transform of the image and computing the slope of the power spectrum.
    Higher frequency components correspond to sharper images so the slope will shift depending on focus quality.

    Parameters:
        image (PIL.Image): Input PIL image.

    Returns:
        float: Focus measure value (slope of the power spectrum).
    """
    starttime = time.perf_counter()

    # Convert to grayscale and convert to numpy array
    img_gray = np.array(image.convert('L'), dtype=np.float32)

    # Apply a Hamming window to reduce edge effects
    hamming_window = hamming(img_gray.shape[0])[:, None] * hamming(img_gray.shape[1])[None, :]
    img_windowed = img_gray * hamming_window

    # Compute the 2D FFT of the image
    fshift = fftshift(fft2(img_windowed))

    # Compute the power spectrum
    power_spectrum = np.abs(fshift) ** 2

    # Create a grid of frequencies and compute the radial profile
    cy, cx = np.array(img_gray.shape) // 2
    y, x = np.indices(img_gray.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    # Radially average the power spectrum
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)  # Avoid division by zero

    # Ignore zero frequencies
    radial_profile = radial_profile[1:]
    frequencies = np.arange(1, len(radial_profile) + 1)

    # Convert to log-log scale
    log_freq = np.log(frequencies)
    log_power = np.log(radial_profile + 1e-8)  # Avoid log(0)

    # Define a linear function for fitting
    def linear_func(x, a, b):
        return a * x + b

    # Fit a line to the log-log power spectrum
    try:
        slope, _ = curve_fit(linear_func, log_freq, log_power)[0]
    except RuntimeError:
        slope = -10.0  # Poor focus

    # The slope of the line is the focus measure
    focus_value = -slope  # Higher values indicate better focus
    print(f"Focus Value: {focus_value} in {time.perf_counter() - starttime:.4f} seconds")

    return focus_value


def auto_focus(mcm301obj, image_queue, params=None):
    """
    Automatically adjusts the focus by moving the stage along the z-axis and measuring the focus quality at each step.
    
    Args:
        mcm301obj (MCM301): The stage controller object for controlling movement.
        image_queue (queue.Queue): Queue containing images to evaluate focus.
        params (list, optional): Optional parameters for custom focus range and steps. If None, uses default configuration.
    """
    # Get the current position of the z-axis (focus axis)
    z = get_pos(mcm301obj, (6,))[0]

    # Adjust the movment distance based on the lens used (large magnification requires smaller steps for focus)
    lens_adjustment_factor = 20 / config.CAMERA_PROPERTIES['lens']

    # Set the range of z-axis positions to test for focus, using custom paramaters if provided
    if params:
        z_range = np.linspace(z - params[0], z + params[0], params[1])
    else:
        z_range = np.linspace(z - int(config.FOCUS_RANGE * lens_adjustment_factor), 
                              z + int(config.FOCUS_RANGE * lens_adjustment_factor), 
                              config.FOCUS_STEPS)
    
    best_z = z  # Store the best z-position for focus
    best_focus = 0  # Store the highest focus value found

    # Loop through each z-position in the range and evaluate the focus
    for z_i in z_range:
        move(mcm301obj, [int(z_i)], (6,)) 
        time.sleep(0.5 * config.CAMERA_PROPERTIES['exposure'] / 1e6)  # Wait for the image to stabilize
        
        # Get the focus value for the current image
        focus = get_focus(image_queue.get(1000))

        # If the current focus is better than the previous best, update the best focus and position
        if focus > best_focus:
            best_focus = focus
            best_z = int(z_i)

    # Move the stage to the best focus position found
    move(mcm301obj, [best_z], (6,))


def initial_auto_focus(gui, mcm301obj, image_queue, n=5):
    """
    Performs a wider initial autofocus procedure.
    
    Args:
        gui: The GUI object to update the interface and control button states.
        mcm301obj (MCM301): The stage controller object for controlling movement.
        image_queue (queue.Queue): Queue containing images to evaluate focus.
        n (int): The number of steps to use for each autofocus iteration (default is 5).
    """
    # Disable the autofocus button in the GUI
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='disabled'))

    # Set the initial large range for focus adjustment, scaled by the camera's lens property
    d = 1e7 / config.CAMERA_PROPERTIES['lens']

    # Continue refining focus until the range is reduced to a certain threshold
    while d > 10e4 / config.CAMERA_PROPERTIES['lens']:
        # Perform autofocus with the current range and steps
        auto_focus(mcm301obj, image_queue, params=[int(d), n])

        # Reduce the focus range for the next iteration
        d = d / (n - 1)

    # Re-enable the autofocus button
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='normal'))



class Monolayer:
    """
    A class that represents a monolayer region, characterized by its contour and image.
    The class also computes various quality metrics.
    
    Args:
        contour (np.ndarray): The contour of the monolayer.
        image (np.ndarray): The image data of the monolayer.
        pos (tuple): The top-left position (x, y) of the monolayer in the global image.
    """
    
    def __init__(self, contour, image, pos):
        self.image = image
        self.global_contour = contour
        x_start, y_start = pos  # The starting position of the contour in the global image
        
        # Calculate the centroid (cx, cy) of the monolayer from the contour
        M = cv2.moments(contour)
        if M['m00'] != 0:  # Avoid division by zero
            self.cx = int(M['m10'] / M['m00'])
            self.cy = int(M['m01'] / M['m00'])
        else:
            # If the area is zero, use the bounding box for centroid calculation
            x, y, w, h = cv2.boundingRect(contour)
            self.cx = x + w // 2
            self.cy = y + h // 2
            
        # Compute area and position-related properties
        self.area_px = cv2.contourArea(contour)
        self.position = (self.cx, self.cy)
        self.area = self.area_px * (config.NM_PER_PX**2)  # Convert area to square nanometers
        self.area_um = self.area / 1e6  # Convert area to square micrometers

        self.contour = contour - np.array([[x_start, y_start]])

        # Compute quality metrics
        self.smoothed_entropy = self.compute_smoothed_entropy()
        self.total_variation_norm = self.compute_tv_norm()
        self.local_intensity_variance = self.compute_local_intensity_variance()
        self.contrast_to_noise_ratio = self.compute_cnr()
        self.skewness = self.compute_skewness()

    def compute_smoothed_entropy(self):
        # Convert image to grayscale if it's not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image  # Assuming the image is already grayscale

        # Apply a light Gaussian blur to reduce pixel noise
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0.5)

        # Create a mask for the contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, thickness=cv2.FILLED)

        # Masked region of interest
        roi = cv2.bitwise_and(smoothed_image, smoothed_image, mask=mask)

        # Compute entropy of the region of interest
        entropy = shannon_entropy(roi)

        return entropy

    def compute_tv_norm(self):
        # Convert image to grayscale if it's not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image  # Assuming the image is already grayscale

        # Apply a light Gaussian blur to reduce pixel noise
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0.5)

        # Create a mask for the contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, thickness=cv2.FILLED)

        # Masked region of interest
        roi = cv2.bitwise_and(smoothed_image, smoothed_image, mask=mask)

        # Compute gradients using Sobel operator
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate total variation norm
        tv_norm = np.sum(np.sqrt(grad_x**2 + grad_y**2))

        # Normalize by the area in pixels to make it size-independent
        tv_norm_normalized = tv_norm / self.area_px if self.area_px != 0 else 0

        return tv_norm_normalized

    def compute_local_intensity_variance(self):
        # Convert image to grayscale if it's not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image  # Assuming the image is already grayscale

        # Create a mask for the contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, thickness=cv2.FILLED)

        # Calculate local variance using a sliding window
        kernel_size = 5
        local_variances = []

        for y in range(0, gray_image.shape[0] - kernel_size, kernel_size):
            for x in range(0, gray_image.shape[1] - kernel_size, kernel_size):
                if mask[y:y+kernel_size, x:x+kernel_size].any():
                    local_patch = gray_image[y:y+kernel_size, x:x+kernel_size]
                    local_variance = np.var(local_patch)
                    local_variances.append(local_variance)

        if local_variances:
            return np.mean(local_variances)  # Mean variance as quality metric
        else:
            return 0  # If no variances are found, return zero

    def compute_cnr(self):
        # Convert image to grayscale if it's not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image  # Assuming the image is already grayscale

        # Create a mask for the contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, thickness=cv2.FILLED)

        # Calculate mean and standard deviation within the masked region
        mean_val, stddev_val = cv2.meanStdDev(gray_image, mask=mask)

        # Calculate mean intensity outside the contour
        mask_inv = cv2.bitwise_not(mask)
        mean_bg, _ = cv2.meanStdDev(gray_image, mask=mask_inv)

        # Calculate CNR
        cnr = abs(mean_val - mean_bg) / (stddev_val + 1e-10)  # Adding epsilon to avoid division by zero

        return cnr[0][0]
    
    def compute_skewness(self):
        # Convert image to grayscale if it's not already
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.image  # Assuming the image is already grayscale

        # Create a mask for the contour
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, thickness=cv2.FILLED)

        # Masked region of interest
        masked_pixels = gray_image[mask == 255]

        # Calculate median and mean of the intensities
        median_intensity = np.median(masked_pixels)
        mean_intensity = np.mean(masked_pixels)

        # Calculate the difference between median and mean intensity
        median_minus_mean = median_intensity - mean_intensity

        return median_minus_mean

class GUI:
    def __init__(self, root, camera):
        # Initialize the main window
        self.root = root
        self.root.title('Camera Control Interface')
        self.root.configure(bg=config.THEME_COLOR)


        # Initialize the stage controller
        self.stage_controller = stage_setup(home=False)

        # Initialize camera and image acquisition thread
        self.camera = camera
        self.image_acquisition_thread = ImageAcquisitionThread(self.camera, rotation_angle=270)
        self.image_acquisition_thread.start()

        self.frame_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)

        self.image_references = []


        # Create main frame with two columns
        self.main_frame = tk.Frame(self.root, bg=config.THEME_COLOR)
        self.main_frame.pack(expand=True, fill="both")

        # Left frame: Live camera view and position labels
        self.left_frame = tk.Frame(self.main_frame, bg=config.THEME_COLOR)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        # Right frame: Notebook with tabs
        self.right_frame = tk.Frame(self.main_frame, bg=config.THEME_COLOR)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=2)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Setup notebook and tabs in the right frame
        self.setup_notebook()

        # Live camera view in the left frame
        self.live_view_canvas = LiveViewCanvas(self.left_frame, self.image_acquisition_thread._image_queue)
        self.live_view_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.live_view_canvas.bind("<Button-1>", self.move_on_click)

        # Initialize live position labels in the left frame
        self.position_frame = tk.Frame(self.left_frame, bg=config.THEME_COLOR)
        self.position_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.init_position_labels()

        # Start live position updates
        self.update_positions()

        # Create control buttons and other UI elements
        self.create_control_buttons()

        # Initialize progress bar and status label
        self.init_progress_bar()
        self.init_main_buttons()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def move_on_click(self, event):
        """Handle a mouse click to move the image."""
        # Access the LiveViewCanvas instance
        canvas = self.live_view_canvas

        # Check if the image attributes are available
        if not all(hasattr(canvas, attr) for attr in ['original_image_width', 'original_image_height', 'displayed_image_width', 'displayed_image_height', 'image_x0', 'image_y0', 'scale']):
            print("Image dimensions not available.")
            return

        # Get the position of the click relative to the canvas
        click_x_canvas, click_y_canvas = event.x, event.y

        # Retrieve the stored image position and scaling
        image_x0 = canvas.image_x0
        image_y0 = canvas.image_y0
        scale = canvas.scale

        # Check if the click is within the displayed image
        if not (image_x0 <= click_x_canvas <= image_x0 + canvas.displayed_image_width and
                image_y0 <= click_y_canvas <= image_y0 + canvas.displayed_image_height):
            print("Click is outside the image.")
            return

        # Map the click position to the image coordinates
        click_x_image = (click_x_canvas - image_x0) / scale
        click_y_image = (click_y_canvas - image_y0) / scale

        # Calculate the center of the image in image coordinates
        image_center_x = canvas.original_image_width / 2
        image_center_y = canvas.original_image_height / 2

        # Calculate the distance and direction from the center of the image
        move_x = click_x_image - image_center_x
        move_y = click_y_image - image_center_y

        # Convert the movement from pixels to nanometers (or appropriate units)
        stage_move_x = move_x * config.NM_PER_PX
        stage_move_y = move_y * config.NM_PER_PX

        movement_array = [stage_move_x, stage_move_y, -stage_move_x, -stage_move_y]

        # Move the stage accordingly
        move_relative(self.stage_controller, [movement_array[(-config.VIEW_ROTATION)%4], movement_array[(-config.VIEW_ROTATION+1)%4]], wait=False)



    def init_progress_bar(self):
        """Initialize the progress bar and status label in the main tab."""
        self.progress_value = tk.DoubleVar()
        self.progress_status = tk.StringVar()
        self.progress_status.set("Idle...")

        # Progress bar widget
        self.progress_bar = ttk.Progressbar(
            self.main_frame_controls,
            orient="horizontal",
            length=200,
            mode="determinate",
            variable=self.progress_value
        )
        self.progress_bar.grid(row=10, column=0, pady=10, sticky='ew', columnspan=2)

        # Progress status label
        self.progress_label = tk.Label(
            self.main_frame_controls,
            textvariable=self.progress_status,
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        self.progress_label.grid(row=11, column=0, pady=5, sticky='w', columnspan=2)

    def init_main_buttons(self):
        """Initialize the buttons in the main tab."""
        # Frame for the buttons
        scan_area_frame = tk.Frame(self.main_frame_controls, bg=config.THEME_COLOR)
        scan_area_frame.grid(row=0, column=0, pady=10, sticky='w', columnspan=2)

        

        # Corner 2 Button
        corner2_button = tk.Button(
            scan_area_frame,
            text="Top Left",
            command=self.update_corner2_position,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        corner2_button.pack(side=tk.LEFT, padx=10)

        # Corner 1 Button
        corner1_button = tk.Button(
            scan_area_frame,
            text="Bottom Right",
            command=self.update_corner1_position,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        corner1_button.pack(side=tk.LEFT, padx=10)

        # Labels to display the positions of Corner 1 and Corner 2
        self.corner1_position_label = tk.Label(
            self.main_frame_controls,
            text="Bottom Right: Not Set",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        self.corner1_position_label.grid(row=2, column=0, pady=5, sticky='w', columnspan=2)

        self.corner2_position_label = tk.Label(
            self.main_frame_controls,
            text="Top Left: Not Set",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        self.corner2_position_label.grid(row=1, column=0, pady=5, sticky='w', columnspan=2)

        # Begin Search Button
        begin_search_button = tk.Button(
            scan_area_frame,
            text="Begin Search",
            command=lambda: run_sequence(
                self,
                self.stage_controller,
                self.image_acquisition_thread._image_queue,
                self.frame_queue,
                config.start_pos,
                config.end_pos,
                self.stitched_view_canvas
            ),
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        begin_search_button.pack(side=tk.BOTTOM, padx=10, pady=10)

    def update_corner1_position(self):
        """Update Corner 1 position label."""
        position = get_pos(self.stage_controller, (4, 5))
        config.corner1.__setitem__(slice(None), position)
        formatted_position = [f"{p:.2e}" for p in position]
        self.corner1_position_label.config(text=f"Bottom Right: {formatted_position}")
        self.update_start_end(config.corner1, config.corner2)

    def update_corner2_position(self):
        """Update Corner 2 position label."""
        position = get_pos(self.stage_controller, (4, 5))
        config.corner2.__setitem__(slice(None), position)
        formatted_position = [f"{p:.2e}" for p in position]
        self.corner2_position_label.config(text=f"Top Left: {formatted_position}")
        self.update_start_end(config.corner1, config.corner2)

    def update_start_end(self, start, end):
        x_1, y_1 = start
        x_2, y_2 = end
        config.start_pos = [min(x_1, x_2), min(y_1, y_2)]
        config.end_pos = [max(x_1, x_2), max(y_1, y_2)]

    def toggle_buttons(self, widget, state: str, exclude=[]):
        """
        Enable or disable all buttons in the main tab, including nested children.

        Args:
            widget: The parent widget containing the buttons.
            state (str): The state of the buttons ('normal' to enable, 'disabled' to disable).
        """
        def recursive_toggle(w):
            """Recursively toggle buttons in the widget and its children."""
            for child in w.winfo_children():
                if child not in exclude:
                    if isinstance(child, tk.Button):
                        child.config(state=state)
                    recursive_toggle(child)
        recursive_toggle(widget)

    def update_progress(self, value, status_text):
        """Update the progress bar value and the status label."""
        self.progress_value.set(value)
        self.progress_status.set(status_text)

    def reset_progress(self):
        """Reset the progress bar and status label to initial values."""
        self.progress_value.set(0)
        self.progress_status.set("Idle...")

    def setup_notebook(self):
        """Setup Notebook and tabs for the GUI."""
        notebook = ttk.Notebook(self.right_frame)
        notebook.pack(expand=True, fill="both")

        self.tabs = {
            "main": ttk.Frame(notebook, style='TFrame'),
            "calibration": ttk.Frame(notebook, style='TFrame'),
            "results": ttk.Frame(notebook, style='TFrame'),
            "plots": ttk.Frame(notebook, style='TFrame')
        }

        notebook.add(self.tabs["main"], text="Main Control")
        notebook.add(self.tabs["calibration"], text="Calibration")
        notebook.add(self.tabs["results"], text="Results & Analysis")
        notebook.add(self.tabs["plots"], text="Plots")

        # Apply styles
        style = ttk.Style()
        style.theme_use('default')

        # Configure styles for the notebook
        style.configure('TFrame', background=config.THEME_COLOR)
        style.configure('TNotebook', background=config.THEME_COLOR, foreground=config.TEXT_COLOR)
        style.configure(
            'TNotebook.Tab',
            background=config.BUTTON_COLOR,
            foreground=config.TEXT_COLOR,
            font=config.LABEL_FONT,
            padding=(5, 1)
        )
        style.map(
            'TNotebook.Tab',
            background=[('selected', config.HIGHLIGHT_COLOR)],
            foreground=[('selected', config.TEXT_COLOR)],
            expand=[('selected', [1, 1, 1, 0])]
        )
        style.configure("TProgressbar", troughcolor=config.THEME_COLOR, background=config.HIGHLIGHT_COLOR, thickness=20)

        # Main Control Tab Layout
        self.main_frame_controls = tk.Frame(self.tabs["main"], bg=config.THEME_COLOR)
        self.main_frame_controls.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Create stitched view canvas
        self.stitched_view_canvas = LiveViewCanvas(self.tabs["main"], queue.Queue(maxsize=config.MAX_QUEUE_SIZE))
        self.stitched_view_canvas.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Configure grid in 'main' tab
        self.tabs["main"].grid_columnconfigure(0, weight=1)
        self.tabs["main"].grid_columnconfigure(1, weight=1)
        self.tabs["main"].grid_rowconfigure(0, weight=1)

        # Calibration Tab Layout
        self.calibration_frame_controls = tk.Frame(self.tabs["calibration"], bg=config.THEME_COLOR)
        self.calibration_frame_controls.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.calibration_frame_controls.columnconfigure(0, weight=1)
        self.calibration_frame_controls.rowconfigure(0, weight=1)

        # Now create gain and exposure controls using grid layout
        self.create_gain_exposure_controls()

        # Create rotation wheel
        self.create_rotation_wheel()

        self.create_color_picker()

        # Results Tab Layout
        self.results_frame_image = tk.Canvas(self.tabs["results"], bg=config.THEME_COLOR, highlightthickness=0)
        self.results_frame_image.pack(fill="both", expand=True, padx=10, pady=10)

        # Results Tab Layout
        self.plots_frame = tk.Canvas(self.tabs["plots"], bg=config.THEME_COLOR, highlightthickness=0)
        self.plots_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Load and display the image
        self.display_results_tab(Image.open("assets/placeholder.webp"))

        # Additional widgets for the results tab
        self.results_frame_controls = tk.Frame(self.tabs["results"], bg=config.THEME_COLOR)
        self.results_frame_controls.pack(fill="both", expand=True, padx=10, pady=10)


    def display_results_tab(self, image, monolayers=None):
        """
        Load and display an image in the left half of the results tab. The image is scaled to fit the Canvas.
        If monolayers is provided, display a treeview on the right half with monolayer images and attributes.
        """
        if not isinstance(image, Image.Image):
            print(f"Invalid image type: {type(image)}. Expected PIL.Image.")
            return

        # Update the image in the left canvas
        MIN_WIDTH = 400  # Set your minimum width here
        MIN_HEIGHT = 400  # Set your minimum height here
        
        # Set minimum size and disable shrinking below this size
        self.results_frame_image.config(width=MIN_WIDTH, height=MIN_HEIGHT)
        self.results_frame_image.grid_propagate(False)  # Prevent the canvas from shrinking below its minimum size

        self.results_frame_image.rowconfigure(0, weight=1) 
        self.results_frame_image.columnconfigure(0, weight=1)

        rotation = view_num_to_rotation[config.VIEW_ROTATION % 4]
        if rotation is not None:
            image = image.transpose(rotation)
        canvas = CanvasImage(self.results_frame_image, image, self)
        canvas.grid(row=0, column=0)

        if monolayers is not None:
            # Clear existing widgets
            for widget in self.results_frame_controls.winfo_children():
                widget.destroy()

            # Define columns
            columns = ('Area (um^2)', 'Centre', 'Entropy', 'TV Norm', 'Intensity Variance', 'CNR', 'Skewness', 'Index')
            visible_rows = 3 # Increased number of visible rows for taller Treeview

            # Create a frame to hold the Treeview and scrollbar
            tree_frame = tk.Frame(self.results_frame_controls, bg=config.THEME_COLOR)
            tree_frame.grid(row=2, column=0, sticky='nsew')

            # Set a fixed height for the frame
            fixed_height = 750  # Increased height to make Treeview taller
            tree_frame.config(height=fixed_height)
            tree_frame.grid_propagate(False)  # Prevent the frame from resizing based on its content

            # Configure the Treeview style for increased row height and custom colors
            style = ttk.Style()

            # Define a unique style name for the Treeview
            tree_style = "Custom.Treeview"

            # Configure the style
            style.configure(tree_style,
                            background=config.THEME_COLOR,  # Background color of the Treeview
                            foreground=config.TEXT_COLOR,  # Text color
                            rowheight=110,  # Row height to accommodate 100px images plus padding
                            fieldbackground=config.THEME_COLOR)  # Background color of fields

            # Configure the headings
            style.configure(f"{tree_style}.Heading",
                            background=config.BUTTON_COLOR,  # Heading background
                            foreground=config.TEXT_COLOR,  # Heading text color
                            font=config.BUTTON_FONT  # Heading font
            )

            # Configure the selected item appearance
            style.map(tree_style,
                    background=[('selected', '#693900')],  # Selection background color
                    foreground=[('selected', config.TEXT_COLOR)])  # Selection text color

            # Create the Treeview with a fixed height and custom style
            tree = ttk.Treeview(
                tree_frame,
                columns=columns,
                show='tree headings',
                height=visible_rows,  # Set the number of visible rows
                style=tree_style  # Apply the custom style
            )

            # Setup column headings
            tree.heading('#0', text='Image')  # Renamed for clarity
            tree.heading('Area (um^2)', text='Area (um^2)')
            tree.heading('Centre', text='Centre')
            tree.heading('Entropy', text='Entropy')
            tree.heading('TV Norm', text='TV Norm')
            tree.heading('Intensity Variance', text='Intensity Variance')
            tree.heading('CNR', text='CNR')
            tree.heading('Skewness', text='Skewness')
            tree.heading('Index', text='Index')

            # Define column widths and alignment
            tree.column('#0', width=120, anchor='center')  # Increased width to 120
            tree.column('Area (um^2)', width=100, anchor='center')
            tree.column('Centre', width=100, anchor='center')
            tree.column('Entropy', width=100, anchor='center')
            tree.column('TV Norm', width=100, anchor='center')
            tree.column('Intensity Variance', width=150, anchor='center')
            tree.column('CNR', width=100, anchor='center')
            tree.column('Skewness', width=100, anchor='center')
            tree.column('Index', width=0, stretch=False)  # Hide the index column

            # Create a vertical scrollbar
            vsb = ttk.Scrollbar(
                tree_frame,
                orient="vertical",
                command=tree.yview
            )
            tree.configure(yscrollcommand=vsb.set)

            # Pack the Treeview and scrollbar inside the frame
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)

            # Insert data into Treeview
            for index, monolayer in enumerate(monolayers):
                img_array = monolayer.image
                if img_array.shape[2] == 4:
                    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA))
                else:
                    img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                img = img.resize((100, 100), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_references.append(photo)

                tree.insert("", "end", image=photo, values=(
                    f"{monolayer.area_um:.2f}",
                    f"{monolayer.position}",
                    f"{monolayer.smoothed_entropy:.2f}",
                    f"{monolayer.total_variation_norm:.2f}",
                    f"{monolayer.local_intensity_variance:.2f}",
                    f"{monolayer.contrast_to_noise_ratio:.2f}",
                    f"{monolayer.skewness:.2f}",
                    index
                ))

            # Bind selection event
            def on_item_selected(event):
                selected_items = tree.selection()
                if not selected_items:
                    return
                selected_item = selected_items[0]
                monolayer_index = int(tree.item(selected_item, 'values')[7])
                monolayer = monolayers[monolayer_index]
                move(self.stage_controller, image_to_stage(monolayer.position, config.start_pos), wait=False)
                print(f"Selected monolayer at index {monolayer_index}, position {monolayer.position}")

            tree.bind('<<TreeviewSelect>>', on_item_selected)

            self.redo_button = tk.Button(
                self.results_frame_controls,
                text="Redo output with new config",
                command=lambda: threading.Thread(
                    target=post_processing,
                    args=(self, config.canvas),
                    daemon=True
                ).start(),
                bg=config.BUTTON_COLOR,
                fg=config.TEXT_COLOR,
                font=config.BUTTON_FONT
            )
            self.redo_button.grid(row=1, column=0, sticky='w', padx=10, pady=10)

    def init_position_labels(self):
        """Initialize position labels for live updates."""
        self.position_names = ["Pos X", "Pos Y", "Focus Z"]
        self.position_labels = []

        for i, name in enumerate(self.position_names):
            label = tk.Label(
                self.position_frame,
                text=name,
                padx=10,
                pady=5,
                bg=config.THEME_COLOR,
                fg=config.TEXT_COLOR,
                font=config.LABEL_FONT
            )
            label.grid(row=i, column=0, sticky='w')
            pos_label = tk.Label(
                self.position_frame,
                text="0.00 nm",
                padx=5,
                bg=config.BUTTON_COLOR,
                fg=config.TEXT_COLOR,
                width=15,
                font=config.LABEL_FONT
            )
            pos_label.grid(row=i, column=1, sticky='w')
            self.position_labels.append(pos_label)


    def update_positions(self):
        """Update the positions displayed in the GUI."""
        positions = get_pos(self.stage_controller, stages=(4, 5, 6))
        for i, pos_label in enumerate(self.position_labels):
            pos_label.config(text=f"{positions[i]:.2e} nm")

        # Schedule the next update
        self.root.after(config.POSITION_UPDATE_INTERVAL, self.update_positions)


    def create_control_buttons(self):
        """Create control buttons and labels for the main and calibration tabs."""
        # Main tab position entry label
        self.position_entry_label = tk.Label(
            self.main_frame_controls,
            text="Enter Positions (nm):",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        self.position_entry_label.grid(row=16, column=0, pady=(10, 0), sticky='w')

        self.create_position_entries()
        self.create_main_frame_buttons()
        self.create_controls()
        self.create_lens_selector()
        self.create_auto_focus()

        advanced_settings_button = tk.Button(
            self.calibration_frame_controls,
            text="Advanced Settings",
            command=self.open_advanced_settings,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        advanced_settings_button.grid(row=16, column=0, padx=10, pady=10, sticky='w')

        self.cat_button = tk.Button(
            self.main_frame_controls,
            text="CAT ATTACK!",
            bg=config.THEME_COLOR,
            bd=0,
            fg=config.THEME_COLOR,
            activebackground=config.HIGHLIGHT_COLOR,
            activeforeground=config.THEME_COLOR,
            font=config.LABEL_FONT,
            command=self.show_cat_image,
        )
        self.cat_button.grid(row=99, column=0, sticky='se')


    def show_cat_image(self):
        cat_image = Image.open(f"assets/cat{random.randint(0,5)}.webp")  # Replace with your cat image path
        cat_image = cat_image.resize((300, 300))  # Resize the image if needed
        
        cat_photo = ImageTk.PhotoImage(cat_image)

        self.cat_button.grid_forget()

        self.cat_canvas = tk.Canvas(self.main_frame_controls, width=300, height=300, bg=config.THEME_COLOR, highlightthickness=0)
        self.cat_canvas.grid(row=99, column=0, sticky='se')

        self.cat_canvas.create_image(0, 0, anchor=tk.NW, image=cat_photo)
        self.cat_canvas.image = cat_photo 
        self.root.after(random.randint(2e3, 15e3), self.restore_cat)

    def restore_cat(self):
        self.cat_button.grid(row=99, column=0, sticky='se')
        self.cat_canvas.grid_forget()

    def create_position_entries(self):
        """Create position entry fields for X, Y, and Z axes, with labels and even spacing."""
        # Create a frame to hold the entries and labels
        position_entry_frame = tk.Frame(self.main_frame_controls, bg=config.THEME_COLOR)
        position_entry_frame.grid(row=17, column=0, pady=(5, 10), padx=10, sticky='w')

        # Define labels and entries for X, Y, Z
        axes = ['X', 'Y', 'Z']
        self.position_entries = {}  # Dictionary to hold entries

        for i, axis in enumerate(axes):
            # Create a sub-frame for each axis to manage alignment
            axis_frame = tk.Frame(position_entry_frame, bg=config.THEME_COLOR)
            axis_frame.pack(side=tk.LEFT, padx=5)

            # Create label
            label = tk.Label(
                axis_frame,
                text=f"{axis}:",
                bg=config.THEME_COLOR,
                fg=config.TEXT_COLOR,
                font=config.LABEL_FONT
            )
            label.pack(side=tk.TOP, anchor='w')

            # Create entry
            entry = tk.Entry(axis_frame, font=config.LABEL_FONT, width=10)
            entry.pack(side=tk.TOP, pady=2)
            entry.bind('<Return>', lambda event, axis=axis: self.submit_entry(event, axis))

            # Store entry in dictionary
            self.position_entries[axis] = entry

        # Clear text in entries when run
        for entry in self.position_entries.values():
            entry.delete(0, tk.END)


    def create_main_frame_buttons(self):
        """Create buttons in the main frame."""
        self.main_frame_controls_position = tk.Frame(self.main_frame_controls, bg=config.THEME_COLOR, padx=25)
        self.main_frame_controls_position.grid(row=0, sticky='w', pady=30)

    def create_controls(self):
        """Create the calibration controls including navigation buttons with equal size."""
        # Insert a spacer frame to add a gap above the move controls
        spacer_frame = tk.Frame(self.main_frame_controls, height=20, bg=config.THEME_COLOR)
        spacer_frame.grid(row=12, column=0, sticky='ew')

        # Label for Move Controls
        move_controls_label = tk.Label(
            self.main_frame_controls,
            text="Move Controls",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.HEADING_FONT
        )
        move_controls_label.grid(row=13, column=0, pady=(10, 0), sticky='w')

        # Create a frame for the navigation and focus buttons
        navigation_frame = tk.Frame(self.main_frame_controls, bg=config.THEME_COLOR)
        navigation_frame.grid(row=14, column=0, padx=10, pady=10, sticky='nsew')

        # Limit the width of the navigation_frame
        navigation_frame.config(width=200)

        # Configure the grid within navigation_frame (3 columns for navigation buttons)
        for col in range(3):
            navigation_frame.columnconfigure(col, weight=1)
        for row in range(3):
            navigation_frame.rowconfigure(row, weight=1)

        stage_a = 4 if config.VIEW_ROTATION % 2 else 5
        stage_b = 5 if config.VIEW_ROTATION % 2 else 4

        dirs = [[-1, -1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [-1, 1, -1, 1]]
        dir = dirs[config.VIEW_ROTATION%4]


        # Navigation buttons positions
        navigation_buttons = [
            ("Up", (0, 1), lambda: move_relative(self.stage_controller, pos=[int(dir[0]*config.DIST)], stages=(stage_a,), wait=False)),
            ("Left", (1, 0), lambda: move_relative(self.stage_controller, pos=[int(dir[1]*config.DIST)], stages=(stage_b,), wait=False)),
            ("Right", (1, 2), lambda: move_relative(self.stage_controller, pos=[int(dir[2]*config.DIST)], stages=(stage_b,), wait=False)),
            ("Down", (2, 1), lambda: move_relative(self.stage_controller, pos=[int(dir[3]*config.DIST)], stages=(stage_a,), wait=False)),
        ]

        # Create navigation buttons with equal size
        for text, (row, col), cmd in navigation_buttons:
            button = tk.Button(
                navigation_frame,
                text=text,
                command=cmd,
                bg=config.BUTTON_COLOR,
                fg=config.TEXT_COLOR,
                font=config.BUTTON_FONT
            )
            button.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')

        # Focus buttons positions (placed below the navigation buttons)
        focus_frame = tk.Frame(self.main_frame_controls, bg=config.THEME_COLOR)
        focus_frame.grid(row=15, column=0, padx=10, pady=(0, 10), sticky='nsew')

        # Configure the grid within focus_frame
        focus_frame.columnconfigure(0, weight=1)
        focus_frame.columnconfigure(1, weight=1)

        focus_buttons = [
            ("Focus +", 0, lambda: move_relative(self.stage_controller, pos=[int(200000/config.CAMERA_PROPERTIES['lens'])], stages=(6,), wait=False)),
            ("Focus -", 1, lambda: move_relative(self.stage_controller, pos=[-int(200000/config.CAMERA_PROPERTIES['lens'])], stages=(6,), wait=False)),
        ]

        for text, col, cmd in focus_buttons:
            button = tk.Button(
                focus_frame,
                text=text,
                command=cmd,
                bg=config.BUTTON_COLOR,
                fg=config.TEXT_COLOR,
                font=config.BUTTON_FONT
            )
            button.grid(row=0, column=col, padx=2, pady=2, sticky='nsew')

    def create_gain_exposure_controls(self):
        """Create Gain and Exposure entry boxes and Auto Expose button in the calibration tab."""

        # Create a frame to hold Gain and Exposure controls
        gain_exposure_frame = tk.Frame(self.calibration_frame_controls, bg=config.THEME_COLOR)
        gain_exposure_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        # Configure grid in gain_exposure_frame
        gain_exposure_frame.columnconfigure(0, weight=1)
        gain_exposure_frame.columnconfigure(1, weight=1)
        gain_exposure_frame.columnconfigure(2, weight=1)

        # Gain Label and Entry
        gain_label = tk.Label(
            gain_exposure_frame,
            text="Gain: (dB)",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        gain_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.gain_entry = tk.Entry(
            gain_exposure_frame,
            font=config.LABEL_FONT,
            width=10
        )
        self.gain_entry.grid(row=0, column=1, padx=5, pady=5)
        self.gain_entry.insert(0, str(config.CAMERA_PROPERTIES['gain']/10))
        self.gain_entry.bind('<Return>', self.update_gain)

        # Exposure Label and Entry
        exposure_label = tk.Label(
            gain_exposure_frame,
            text="Exposure: (ms)",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        exposure_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.exposure_entry = tk.Entry(
            gain_exposure_frame,
            font=config.LABEL_FONT,
            width=10
        )
        self.exposure_entry.grid(row=1, column=1, padx=5, pady=5)
        self.exposure_entry.insert(0, str(config.CAMERA_PROPERTIES['exposure']/1000))
        self.exposure_entry.bind('<Return>', self.update_exposure)

        # Auto Expose Button
        self.auto_expose_button = tk.Button(
            gain_exposure_frame,
            text="Auto Expose",
            command=self.auto_camera_settings,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        self.auto_expose_button.grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky='ns')

        # Software Exposure Increase Checkbox
        self.software_exposure_var = tk.BooleanVar(value=config.BINNING_EXPOSURE_INCREASE)
        self.software_exposure_checkbox = tk.Checkbutton(
            gain_exposure_frame,
            text="Software Exposure Increase",
            variable=self.software_exposure_var,
            command=self.update_software_exposure_increase,
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR, 
            font=config.LABEL_FONT,
            activebackground=config.THEME_COLOR,
            selectcolor=config.THEME_COLOR,
        )
        self.software_exposure_checkbox.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky='w')

    def update_software_exposure_increase(self):
        """Update the BINNING_EXPOSURE_INCREASE configuration based on the checkbox state."""
        config.BINNING_EXPOSURE_INCREASE = self.software_exposure_var.get()


    def open_advanced_settings(self):
        """Open a window to adjust advanced configuration settings."""
        # Create a new top-level window
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Advanced Settings")
        settings_window.configure(bg=config.THEME_COLOR)
        settings_window.grab_set()  # Make the window modal

        # Define the list of variables to display and edit
        # Each tuple contains (Display Name, Config Attribute Name)
        settings_vars = [
            ("Maximum distance to check for a better focus (nm)", "FOCUS_RANGE"),
            ("Acceptable proportion of being out of focus", "FOCUS_BUFFER"),
            ("Number of frames to avg when assessing for a change in focus", "FOCUS_FRAME_AVG"),
            ("Amount to downscale the results image", "RESULTS_IMAGE_DOWNSCALE"),
            # Add more variables as needed
        ]

        # Dictionary to hold entry widgets for later access
        self.advanced_settings_entries = {}

        # Create labels and entry boxes for each variable
        for i, (label_text, var_name) in enumerate(settings_vars):
            # Label for the variable name
            label = tk.Label(
                settings_window,
                text=label_text + ":",
                bg=config.THEME_COLOR,
                fg=config.TEXT_COLOR,
                font=config.LABEL_FONT
            )
            label.grid(row=i, column=0, padx=10, pady=5, sticky='e')

            # Entry box for the variable value
            entry = tk.Entry(
                settings_window,
                font=config.LABEL_FONT,
                width=20
            )
            entry.grid(row=i, column=1, padx=10, pady=5, sticky='w')

            # Insert the current value from config
            current_value = getattr(config, var_name, "")
            entry.insert(0, str(current_value))

            # Store the entry widget for later retrieval
            self.advanced_settings_entries[var_name] = entry

        # Save and Cancel buttons
        button_frame = tk.Frame(settings_window, bg=config.THEME_COLOR)
        button_frame.grid(row=len(settings_vars), column=0, columnspan=2, pady=10)

        save_button = tk.Button(
            button_frame,
            text="Save",
            command=lambda: self.save_advanced_settings(settings_window, settings_vars),
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        save_button.pack(side=tk.LEFT, padx=5)

        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=settings_window.destroy,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        cancel_button.pack(side=tk.LEFT, padx=5)

    def save_advanced_settings(self, window, settings_vars):
        """Save the updated settings from the advanced settings window."""
        for label_text, var_name in settings_vars:
            entry = self.advanced_settings_entries.get(var_name)
            if entry:
                new_value = entry.get().strip()
                try:
                    # Determine the type based on the current config value
                    current_value = getattr(config, var_name, None)
                    if isinstance(current_value, int):
                        converted_value = int(new_value)
                    elif isinstance(current_value, float):
                        converted_value = float(new_value)
                    else:
                        converted_value = new_value  # For strings or other types

                    # Optionally, add range checks or other validations here
                    # Example:
                    # if var_name == "FOCUS_RANGE" and not (0 < converted_value < 1000):
                    #     raise ValueError("Focus Range must be between 0 and 1000")

                    # Update the config with the new value
                    setattr(config, var_name, converted_value)
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid value for {label_text}. Please enter a valid number.")
                    return  # Exit the method without closing the window

        # Optionally, update any dependent UI elements or settings here

        # Close the settings window after saving
        window.destroy()
        messagebox.showinfo("Settings Saved", "Advanced settings have been updated successfully.")

    
    def create_lens_selector(self):

        lens_frame = tk.Frame(self.calibration_frame_controls, bg=config.THEME_COLOR)
        lens_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        lens_frame.columnconfigure(0, weight=1)
        lens_frame.columnconfigure(1, weight=1)

        lens_label = tk.Label(
            lens_frame,
            text="Magnification: ",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        lens_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.lens_entry = tk.Entry(
            lens_frame,
            font=config.LABEL_FONT,
            width=10
        )
        self.lens_entry.grid(row=0, column=1, padx=5, pady=5)
        self.lens_entry.insert(0, str(config.CAMERA_PROPERTIES['lens']))
        self.lens_entry.bind('<Return>', self.update_lens)

    def create_auto_focus(self):

        auto_focus_frame = tk.Frame(self.calibration_frame_controls, bg=config.THEME_COLOR)
        auto_focus_frame.grid(row=1, column=3, columnspan=2, padx=10, pady=10, sticky='nsew')

        self.auto_focus_button = tk.Button(
            auto_focus_frame,
            text="Auto Focus",
            command=lambda: threading.Thread(
                    target=initial_auto_focus,
                    args=(self, self.stage_controller, self.image_acquisition_thread._image_queue),
                    daemon=True
                ).start(),
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        self.auto_focus_button.grid(row=0, column=1, padx=5, pady=5)

        self.auto_focus_on = tk.BooleanVar(value=config.AUTOFOCUS) 

        # Checkbutton for enabling/disabling auto focus
        self.auto_focus_checkbox = tk.Checkbutton(
            auto_focus_frame,
            text="Auto Focus Enabled",
            variable=self.auto_focus_on,
            command=self.toggle_auto_focus,
            onvalue=True, 
            offvalue=False, 
            height=2, 
            width=20,
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR, 
            font=config.LABEL_FONT,
            activebackground=config.THEME_COLOR,
            selectcolor=config.THEME_COLOR,
        )
        self.auto_focus_checkbox.grid(row=0, column=0, padx=5, pady=5)

        config.AUTOFOCUS = self.auto_focus_on.get()

    def toggle_auto_focus(self):
        config.AUTOFOCUS = self.auto_focus_on.get()
        print(f"Auto focus set to: {config.AUTOFOCUS}")


    def update_lens(self, event=None):
        try:
            lens_value = float(self.lens_entry.get())
            if lens_value <= 0:
                lens_value = max(1, lens_value)
                self.lens_entry.delete(0, tk.END)
                self.lens_entry.insert(0, str(lens_value))
            config.CAMERA_PROPERTIES['lens'] = lens_value
            config.NM_PER_PX = config.CAMERA_PROPERTIES["px_size"] * config.CAMERA_BINNING**2 * 1000 / config.CAMERA_PROPERTIES["lens"]
            config.DIST = int(min(config.CAMERA_DIMS) * config.NM_PER_PX * (1 - config.IMAGE_OVERLAP))
            print(config.NM_PER_PX)

            print(f"Magnification updated to: {lens_value}")
        except ValueError:
            messagebox.showerror("Input Error", "Magnification must be a number!")

    def update_gain(self, event=None):
        """Update the gain value from the entry box."""
        try:
            gain_value = float(self.gain_entry.get())
            if gain_value < 0 or gain_value > 48:
                gain_value = min(max(0, gain_value), 48)
                self.gain_entry.delete(0, tk.END)
                self.gain_entry.insert(0, str(gain_value))
            config.CAMERA_PROPERTIES['gain'] = int(gain_value*10)
            self.camera.gain = config.CAMERA_PROPERTIES["gain"]
            self.camera.exposure_time_us = config.CAMERA_PROPERTIES["exposure"]

            print(f"Gain updated to: {gain_value}")
        except ValueError:
            messagebox.showerror("Input Error", "Gain must be a number!")

    def update_exposure(self, event=None):
        """Update the exposure value from the entry box."""
        try:
            exposure_value = float(self.exposure_entry.get())
            if exposure_value < 0 or exposure_value > 30000:
                exposure_value = min(max(0, exposure_value), 30000)
                self.exposure_entry.delete(0, tk.END)
                self.exposure_entry.insert(0, str(exposure_value))
            config.CAMERA_PROPERTIES['exposure'] = int(exposure_value*1000)
            self.camera.gain = config.CAMERA_PROPERTIES["gain"]
            self.camera.exposure_time_us = config.CAMERA_PROPERTIES["exposure"]
            
            print(f"Exposure updated to: {exposure_value}")
        except ValueError:
            messagebox.showerror("Input Error", "Exposure must be a number!")

    def auto_camera_settings(self):
        """Auto camera calibration to run in a separate thread."""
        def run_auto_exposure():
            """Threaded auto-exposure function to adjust gain and exposure."""
            # Disable the button while auto-calibration is running
            self.auto_expose_button.config(state='disabled')

            test_gain, test_exposure = 100, 33000
            max_gain = 350
            config.BINNING_EXPOSURE_INCREASE = False
            self.software_exposure_var.set(config.BINNING_EXPOSURE_INCREASE)
            target = (config.BRIGHTNESS_RANGE[0] + config.BRIGHTNESS_RANGE[1]) / 2
            self.camera.gain = int(test_gain)
            self.camera.exposure_time_us = int(test_exposure)
            time.sleep(2*test_exposure/1e6)
            for i in range(3):
                latest_frame = np.array(self.image_acquisition_thread._image_queue.get(max(int(test_exposure/1e3), 1000)))
            average_intensity = latest_frame.mean()
            
            while not (config.BRIGHTNESS_RANGE[0] <= average_intensity <= config.BRIGHTNESS_RANGE[1]):
                delta_intensity = target - average_intensity
                
                if 1 < test_gain < max_gain:
                    if average_intensity > 250 and test_gain > 5 :
                        test_gain *= 0.5
                    elif average_intensity < 10:
                        test_gain *= 1.99
                    elif average_intensity < config.BRIGHTNESS_RANGE[0]:
                        test_gain *= min(config.BRIGHTNESS_RANGE[0]/average_intensity, 1.2)
                    elif average_intensity > config.BRIGHTNESS_RANGE[1] and test_gain > 5:
                        test_gain *= min(config.BRIGHTNESS_RANGE[1]/average_intensity, 1.2)
                    else:
                        test_gain += delta_intensity/10
                    test_gain = min(max(1, test_gain), max_gain)
                else:
                    if test_exposure > 200000 and not config.BINNING_EXPOSURE_INCREASE:
                        config.BINNING_EXPOSURE_INCREASE = True
                        self.software_exposure_var.set(config.BINNING_EXPOSURE_INCREASE)
                        test_gain, test_exposure = 100, 33000
                        max_gain = 200
                    elif average_intensity > 240:
                        test_exposure *= 0.5
                    elif average_intensity < 16:
                        test_exposure *= 1.99
                    elif average_intensity < config.BRIGHTNESS_RANGE[0]:
                        test_exposure *= min(config.BRIGHTNESS_RANGE[0]/average_intensity, 1.2)
                    elif average_intensity > config.BRIGHTNESS_RANGE[1]:
                        test_exposure *= min(config.BRIGHTNESS_RANGE[1]/average_intensity, 1.2)
                    else: 
                        test_exposure += delta_intensity*1000
                    test_exposure = min(max(10, test_exposure), 30000000)

                self.camera.gain = int(test_gain)
                self.camera.exposure_time_us = int(test_exposure)

                # Get the next frame and update the average intensity
                # time.sleep(2*test_exposure/1e6)
                latest_frame = np.array(self.image_acquisition_thread._image_queue.get(max(int(test_exposure/1e3), 1000)))
                average_intensity = latest_frame.mean()
                print(average_intensity)
                print(f"Gain: {test_gain}, Exposure: {test_exposure}, Bin Sum:{config.BINNING_EXPOSURE_INCREASE}")

            # Update config with final gain and exposure values
            config.CAMERA_PROPERTIES['gain'] = int(test_gain)
            config.CAMERA_PROPERTIES['exposure'] = int(test_exposure)

            # Update GUI entries
            self.gain_entry.delete(0, tk.END)
            self.gain_entry.insert(0, str(config.CAMERA_PROPERTIES['gain'] / 10))
            self.exposure_entry.delete(0, tk.END)
            self.exposure_entry.insert(0, str(config.CAMERA_PROPERTIES['exposure'] / 1000))

            self.software_exposure_var.set(config.BINNING_EXPOSURE_INCREASE)

            # Re-enable the button after the process is complete
            self.auto_expose_button.config(state='normal')

        # Start the auto calibration in a separate thread
        threading.Thread(target=run_auto_exposure, daemon=True).start()

    def create_color_picker(self):
        color_frame = tk.Frame(self.calibration_frame_controls, bg=config.THEME_COLOR)
        color_frame.grid(row=0, column=3)
        
        # Set the default color
        self.color_code = rgb2hex(config.MONOLAYER_COLOR)

        # Add a text label to the left
        self.text_label = tk.Label(
            color_frame,
            text="Fluorescence\n Detection Colour:",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
            )
        self.text_label.grid(row=0, column=0, padx=10)

        self.color_box = tk.Frame(
            color_frame,
            bg=self.color_code,
            width=30,
            height=30, 
            bd=1,
            relief='solid'
        )
        self.color_box.grid(row=0, column=1, padx=5, pady=5)
        self.color_box.pack_propagate(False)

        # Bind the label click event to open the color picker
        self.color_box.bind("<Button-1>", self.pick_color)

    def pick_color(self, event):
        # Open the color picker dialog
        color_code = colorchooser.askcolor(initialcolor=self.color_code, title="Choose a color")

        # If a color is selected, update the color_code and change the background of the label
        if color_code[1]:  # Check if a valid color was selected
            self.color_code = color_code[1] 
            self.color_box.config(bg=self.color_code)
            rgb_tuple = ImageColor.getrgb(self.color_code)
            config.MONOLAYER_COLOR = rgb_tuple


    def create_rotation_wheel(self):
        """Create a 360-degree rotation wheel in the calibration tab."""
        wheel_frame = tk.Frame(self.calibration_frame_controls, bg=config.THEME_COLOR)
        wheel_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

        # Use grid instead of pack for the canvas
        self.canvas = tk.Canvas(wheel_frame, width=100, height=100, bg=config.THEME_COLOR, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.wheel_radius = 40
        self.wheel_center_x = 50
        self.wheel_center_y = 50

        self.wheel_image = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        self.wheel_draw = ImageDraw.Draw(self.wheel_image)
        self.draw_anti_aliased_wheel()

        self.wheel_tk_image = ImageTk.PhotoImage(self.wheel_image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.wheel_tk_image)
        self.pointer_line = self.canvas.create_line(
            self.wheel_center_x,
            self.wheel_center_y,
            self.wheel_center_x + self.wheel_radius,
            self.wheel_center_y,
            width=2,
            fill=config.HIGHLIGHT_COLOR
        )
        self.canvas.bind("<B1-Motion>", self.update_wheel)

        # Use grid for the angle entry
        self.angle_entry = tk.Entry(wheel_frame, width=5, font=config.LABEL_FONT)
        self.angle_entry.grid(row=1, column=0, pady=5)
        self.angle_entry.insert(0, "270")
        self.angle_entry.bind('<Return>', self.set_angle_from_entry)

        # Place the label directly under the entry box
        rotation_wheel_label = tk.Label(
            wheel_frame,
            text="Camera Rotation",
            background=config.THEME_COLOR,
            foreground=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        rotation_wheel_label.grid(row=2, column=0, pady=5, sticky='n')

        end_x = self.wheel_center_x + self.wheel_radius * math.cos(math.radians(270))
        end_y = self.wheel_center_y + self.wheel_radius * math.sin(math.radians(270))
        self.canvas.coords(self.pointer_line, self.wheel_center_x, self.wheel_center_y, end_x, end_y)


    def on_focus_slider_change(self, event):
        """Handle changes in the focus slider."""
        focus_value = int(event)
        print(f"Focus Slider moved to: {focus_value}")

    def draw_anti_aliased_wheel(self):
        """Draw an anti-aliased wheel for the rotation control."""
        self.wheel_draw.ellipse(
            [self.wheel_center_x - self.wheel_radius, self.wheel_center_y - self.wheel_radius,
             self.wheel_center_x + self.wheel_radius, self.wheel_center_y + self.wheel_radius],
            outline=config.HIGHLIGHT_COLOR, width=2
        )

    def update_wheel(self, event):
        """Update the rotation wheel based on mouse movement."""
        dx = event.x - self.wheel_center_x
        dy = event.y - self.wheel_center_y
        angle = int(math.degrees(math.atan2(dy, dx))) % 360

        end_x = self.wheel_center_x + self.wheel_radius * math.cos(math.radians(angle))
        end_y = self.wheel_center_y + self.wheel_radius * math.sin(math.radians(angle))
        self.canvas.coords(self.pointer_line, self.wheel_center_x, self.wheel_center_y, end_x, end_y)

        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{angle:.0f}")
        self.image_acquisition_thread._rotation_angle = float(self.angle_entry.get())

    def submit_entry(self, event, axis):
        """Submit the position entry for a single axis and initiate movement."""
        entry_value = self.position_entries[axis].get().strip()
        if self.validate_entries(entry_value):
            position = int(entry_value)
            if axis in ['X', 'Y']:
                # Map axis to stage number
                stage = 4 if axis == 'X' else 5
                threading.Thread(
                    target=self.move_and_update_progress,
                    args=([position], (stage,)),
                    daemon=True
                ).start()
            elif axis == 'Z':
                threading.Thread(
                    target=self.move_and_update_progress,
                    args=([position], (6,)),
                    daemon=True
                ).start()
            # Clear the entry after submission
            self.position_entries[axis].delete(0, tk.END)

    def validate_entries(self, *entries):
        """Validate that the entries are integers."""
        for entry in entries:
            try:
                int(entry)
            except ValueError:
                messagebox.showerror("Input Error", "All fields must be integers!")
                return False
        return True

    def move_and_update_progress(self, pos, stages=(4, 5)):
        """Move the stage to a position and update the progress bar during the operation."""
        self.update_progress(0, "Moving stage...")

        move(self.stage_controller, pos, stages, wait=False)  # Simulating movement
        self.root.update_idletasks()  # Ensure UI gets updated during the loop

        self.update_progress(100, "Movement complete.")

    def validate_entries(self, *entries):
        """Validate that the entries are integers."""
        for entry in entries:
            if not entry.isdigit():
                messagebox.showerror("Input Error", "All fields must be integers!")
                return False
        return True

    def set_angle_from_entry(self, event):
        """Set the rotation angle from the entry field."""
        try:
            angle = float(self.angle_entry.get()) % 360
        except ValueError:
            angle = 0

        end_x = self.wheel_center_x + self.wheel_radius * math.cos(math.radians(angle))
        end_y = self.wheel_center_y + self.wheel_radius * math.sin(math.radians(angle))
        self.canvas.coords(self.pointer_line, self.wheel_center_x, self.wheel_center_y, end_x, end_y)
        self.image_acquisition_thread._rotation_angle = angle

        print(f"Camera rotation set to: {angle:.2f} degrees")

    def on_closing(self):
        """Handle the cleanup on closing the application."""
        logging.info("GUI is closing. Initiating cleanup.")
        try:
            if self.image_acquisition_thread.is_alive():
                logging.info("Stopping image acquisition thread...")
                self.image_acquisition_thread.stop()
                self.image_acquisition_thread.join()
                logging.info("Image acquisition thread stopped.")
            try:
                if not getattr(self.camera, 'disposed', False):
                    logging.info("Disarming and disposing the camera...")
                    self.camera.disarm()
                    self.camera.dispose()
                    self.camera.disposed = True  # Mark the camera as disposed
            except (TLCameraError, AttributeError) as e:
                logging.exception("Error while disarming or disposing the camera in on_closing:")
        except Exception as e:
            logging.exception("Error during shutdown:")
        self.root.destroy()
        logging.info("GUI closed successfully.")


def run_sequence(gui, mcm301obj, image_queue, frame_queue, start, end, stitched_view_canvas):
    """
    Executes the main scanning sequence, capturing images and stitching them together.
    
    Args:
        gui: The GUI object for controlling the interface.
        mcm301obj (MCM301): The stage controller object for controlling movement.
        image_queue (queue.Queue): Queue containing captured images from the camera.
        frame_queue (queue.Queue): Queue to store images and their positions for stitching.
        start (tuple): Starting coordinates (x, y) in nanometers.
        end (tuple): Ending coordinates (x, y) in nanometers.
        stitched_view_canvas: The canvas where the stitched result will be displayed.
    """
    # Ensure the start and end points are properly ordered
    x_1, y_1 = start
    x_2, y_2 = end
    start = [min(x_1, x_2), min(y_1, y_2)]  # Adjust to get the top-left corner
    end = [max(x_1, x_2), max(y_1, y_2)]  # Adjust to get the bottom-right corner
    
    # Disable buttons in the main tab except a specified button
    gui.root.after(0, lambda: gui.toggle_buttons(gui.main_frame_controls, 'disabled', exclude=[gui.cat_button]))
    gui.root.after(0, lambda: gui.display_results_tab(Image.open("assets/placeholder.webp")))  # Display a placeholder image
    gui.image_references = []  # Clear image references in the GUI

    # Start a thread to handle stitching images in the background
    stitching_thread = threading.Thread(target=stitch_and_display_images, args=(gui, frame_queue, start, end, stitched_view_canvas))
    stitching_thread.start()

    # Start the scanning algorithm in a separate thread
    alg_thread = threading.Thread(target=alg, args=(gui, mcm301obj, image_queue, frame_queue, start, end))
    alg_thread.start()

    # Define a function to re-enable buttons once stitching is complete
    def check_stitching_complete():
        stitching_thread.join()  # Wait for stitching to finish
        gui.root.after(0, lambda: gui.toggle_buttons(gui.main_frame_controls, 'normal'))  # Re-enable buttons after completion

    # Start a daemon thread to check when stitching is complete and re-enable buttons
    threading.Thread(target=check_stitching_complete, daemon=True).start()


def cleanup(signum=None, frame=None):
    """
    Cleans up resources, such as the camera, when a signal is received or the program exits.
    
    Args:
        signum (int, optional): The signal number (e.g., SIGTERM, SIGINT).
        frame (optional): The current stack frame.
    """
    logging.info("Signal received, cleaning up...")

    try:
        # If the GUI is running, close it properly
        if 'GUI_main' in globals() and GUI_main is not None:
            GUI_main.on_closing()
        # If the camera is active and not disposed, disarm and dispose it safely
        elif 'camera' in globals() and camera is not None:
            if not getattr(camera, 'disposed', False):
                try:
                    logging.info("Disarming and disposing the camera...")
                    camera.disarm()
                    camera.dispose()
                    camera.disposed = True  # Mark the camera as disposed
                except (TLCameraError, AttributeError) as e:
                    logging.exception("Error while disarming or disposing the camera during cleanup:")
            else:
                logging.info("Camera already disposed.")
    except Exception as e:
        logging.exception(f"Exception during cleanup: {e}")

    # Exit the program safely
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for unexpected termination
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        logging.info("Initializing TLCameraSDK...")
        with TLCameraSDK() as sdk:
            logging.info("Discovering available cameras...")
            camera_list = sdk.discover_available_cameras()
            logging.debug(f"Available cameras: {camera_list}")
            if len(camera_list) == 0:
                logging.error("No cameras found.")
                sys.exit(1)

            # Open the camera without using 'with' so it stays open
            camera = sdk.open_camera(camera_list[-1])
            logging.info(f"Camera {camera_list[-1]} initialized.")  # Fixed index to match opened camera

            # Ensure the camera is properly disarmed before setting parameters
            if camera.is_armed:
                logging.warning("Camera was already armed. Disarming first.")
                camera.disarm()

            # Set camera parameters before arming
            logging.info("Setting camera parameters...")
            camera.frames_per_trigger_zero_for_unlimited = 1  # Capture one frame per software trigger
            camera.gain = config.CAMERA_PROPERTIES["gain"]
            camera.exposure_time_us = config.CAMERA_PROPERTIES["exposure"]
            logging.debug(f"Camera gain set to {camera.gain}")
            logging.debug(f"Camera exposure time set to {camera.exposure_time_us} us")

            # Arm the camera
            try:
                logging.info("Arming the camera...")
                camera.arm(2)
                logging.debug(f"Camera is_armed: {camera.is_armed}")
            except Exception as e:
                logging.exception("Error arming the camera:")
                camera.disarm()
                camera.dispose()
                sys.exit(1)

            if not camera.is_armed:
                logging.error("Camera failed to arm. Please disconnect and reconnect the camera.")
                camera.dispose()
                sys.exit(1)

            try:
                # Update config parameters
                config.CAMERA_PROPERTIES["px_size"] = (
                    camera.sensor_pixel_height_um + camera.sensor_pixel_width_um
                ) / 2
                config.NM_PER_PX = (
                    config.CAMERA_PROPERTIES["px_size"] * config.CAMERA_BINNING**2 * 1000 / config.CAMERA_PROPERTIES["lens"]
                )

                config.CAMERA_DIMS = [int(camera.image_width_pixels/(config.CAMERA_BINNING**2)), int(camera.image_height_pixels/(config.CAMERA_BINNING**2))]
                print([camera.image_width_pixels, camera.image_height_pixels])
                config.DIST = int(
                    min(config.CAMERA_DIMS) * config.NM_PER_PX * (1 - config.IMAGE_OVERLAP)
                )
                logging.debug(f"Config updated: {config.CAMERA_PROPERTIES}")

                logging.info("App initializing...")
                root = tk.Tk()
                GUI_main = None  # Initialize GUI_main before try-except

                try:
                    GUI_main = GUI(root, camera)
                    root.after(1000, GUI_main.auto_camera_settings)
                    logging.info("App starting")
                    root.mainloop()
                except Exception as e:
                    logging.exception("An exception occurred during GUI initialization or main loop:")
                    # Ensure proper cleanup
                    if GUI_main is not None:
                        GUI_main.on_closing()
                    else:
                        # Attempt to disarm and dispose the camera
                        try:
                            if camera.is_armed:
                                logging.info("Disarming the camera...")
                                camera.disarm()
                        except (TLCameraError, AttributeError):
                            logging.exception("Error while disarming the camera during exception handling:")
                        try:
                            logging.info("Disposing the camera...")
                            camera.dispose()
                        except (TLCameraError, AttributeError):
                            logging.exception("Error while disposing the camera during exception handling:")
                    sys.exit(1)

                logging.info("Waiting for image acquisition thread to finish...")
                if GUI_main and hasattr(GUI_main, 'image_acquisition_thread') and GUI_main.image_acquisition_thread.is_alive():
                    GUI_main.image_acquisition_thread.stop()
                    GUI_main.image_acquisition_thread.join()

                logging.info("Closing resources...")
            except Exception as e:
                logging.exception("An exception occurred during application execution:")
            finally:
                # Ensure the camera is disarmed and disposed
                if not getattr(camera, 'disposed', False):
                    try:
                        logging.info("Disarming and disposing the camera...")
                        camera.binx = 1
                        camera.biny = 1
                        camera.disarm()
                        camera.dispose()
                        camera.disposed = True  # Mark the camera as disposed
                    except (TLCameraError, AttributeError) as e:
                        logging.exception("Error while disarming or disposing the camera during final cleanup:")
                else:
                    logging.info("Camera already disposed.")
                logging.info("App terminated. Goodbye!")

    except Exception as e:
        logging.exception("An error occurred during camera initialization:")
