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
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from scipy.optimize import curve_fit
from scipy.signal.windows import hamming

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw, ImageColor

import cv2  # OpenCV library for accessing the webcam
import numpy as np
from skimage.measure import shannon_entropy
import torch

from custom_definitions import *
from zoom_canvas import *
import config

from MCM301_COMMAND_LIB import *

import logging

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



# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all types of logs
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('app_debug.log', mode='w')  # Log to a file
    ]
)



class LiveViewCanvas(tk.Canvas):
    def __init__(self, parent, image_queue):
        super().__init__(parent, bg=config.THEME_COLOR, highlightthickness=0)
        self.image_queue = image_queue
        self.is_active = True  # Initialize attribute to control image updates
        self.bind("<Configure>", self.on_resize)  # Bind resizing event
        self._display_image()

    def _display_image(self):
        if not self.is_active:
            self.after(config.UPDATE_DELAY, self._display_image)
            return

        image = None
        try:
            # Retrieve all available images and use the latest one
            while True:
                try:
                    image = self.image_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error retrieving image from queue: {e}")

        if image:
            self.update_image_display(image)

        self.after(config.UPDATE_DELAY, self._display_image)  # Continue updating the image

    def update_image_display(self, image):
        # Validate image dimensions
        if image.width <= 0 or image.height <= 0:
            logging.warning(f"Received image with invalid dimensions: {image.width}x{image.height}. Skipping resize.")
            return

        # Store the original image size
        self.original_image_width = image.width
        self.original_image_height = image.height
        self.current_image = image  # Store the current image

        # Get the current size of the canvas
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()

        # Avoid division by zero
        if canvas_width == 0 or canvas_height == 0:
            logging.warning("Canvas width or height is zero. Skipping image display.")
            return

        # Compute the scaling factors to fit the image to the canvas
        scale_x = canvas_width / self.original_image_width
        scale_y = canvas_height / self.original_image_height
        self.scale = min(scale_x, scale_y)

        # Ensure scaling factors are positive
        if self.scale <= 0:
            logging.warning(f"Invalid scaling factors: scale_x={scale_x}, scale_y={scale_y}. Skipping resize.")
            return

        # Resize the image accordingly
        new_width = max(int(self.original_image_width * self.scale), 1)  # Ensure width is at least 1
        new_height = max(int(self.original_image_height * self.scale), 1)  # Ensure height is at least 1
        self.displayed_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Store the displayed image size
        self.displayed_image_width = new_width
        self.displayed_image_height = new_height

        # Clear the canvas
        self.delete("all")

        # Update the image on the canvas
        self.photo_image = ImageTk.PhotoImage(self.displayed_image)
        # Center the image on the canvas
        self.image_x0 = (canvas_width - new_width) // 2
        self.image_y0 = (canvas_height - new_height) // 2
        self.create_image(self.image_x0, self.image_y0, anchor='nw', image=self.photo_image)


    def on_resize(self, event):
        # Redraw the image when the canvas is resized
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

    # Convert the relevant regions to PyTorch tensors for GPU processing
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

    # Start timing right before initializing ThreadPoolExecutor
    t1 = time.time()

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

            scaled_canvas = scale_down_canvas(canvas, 16)

            stitched_image = Image.fromarray(cv2.cvtColor(scaled_canvas, cv2.COLOR_BGRA2RGBA))
            stitched_view_canvas.update_image_display(stitched_image)

        # Wait for all futures to complete
        for future in futures:
            future.result()

    # End timing after all processing is complete
    t2 = time.time()
    print("Image stitching complete")
    print(f"Time taken: {t2 - t1:.2f} seconds")

    print("Saving final image...")
    #cv2.imwrite("Images/final.png", canvas)
    #cv2.imwrite("Images/final_compressed.png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    save_image(canvas, "Images/final_scan.png", 10)

    print("Save complete")

    scaled_canvas = scale_down_canvas(canvas, 16)

    stitched_image = Image.fromarray(cv2.cvtColor(scaled_canvas, cv2.COLOR_BGRA2RGBA))
    stitched_view_canvas.update_image_display(stitched_image)

    config.canvas = canvas

    if config.SAVE_RAW_IMAGE:
        threading.Thread(target=lambda: save_image(canvas, "Images/RAW.png"), daemon=True).start()
        

    post_processing(gui, canvas, start)

def post_processing(gui, canvas, start, contrast=2, threshold=50):
    t1 = time.time()
    gui.root.after(0, lambda: gui.update_progress(95, "Image processing..."))
    print("Post processing...")

    downscale_factor = 4
    monolayers = []

    # Create a downsampled version of the canvas for post-processing
    post_image = scale_down_canvas(canvas, downscale_factor)

    # post_image = cv2.blur(post_image, (5, 5)) # Optional pre grayscale blur (downscaling already blurs)
    # Our image is in BGRA format so to convert it to greyscale with a bias for red and a bias against green and blue, we use the following formula:
    red, green, blue  = [(c/127)-1 for c in config.MONOLAYER_COLOR]
    print(f"{red}, {green}, {blue}")

    post_image = red * post_image[:, :, 2] + green * post_image[:, :, 1] + blue * post_image[:, :, 0]
    # Normalize the image to 0-255
    post_image = np.clip(post_image, 0, 255)
    # Convert to uint8
    post_image = post_image.astype(np.uint8)
    # Apply a light Gaussian blur to reduce pixel noise and prevent against accidental monolayer splitting
    post_image = cv2.blur(post_image, (4, 4))
    
    # Increase contrast
    post_image = cv2.convertScaleAbs(post_image, alpha=contrast, beta=0)

    # Remove pixels below a the threshold value
    _, post_image = cv2.threshold(post_image, threshold, 255, cv2.THRESH_BINARY)

    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")


    # print("Saving post processed image...")
    # save_image(post_image, "Images/processed.png")
    # print("Saved!")

    # Create a copy of the canvas for contour drawing (this is slow but necessary if canvas is to be used again in future)
    contour_image = canvas.copy()

    print("Locating Monolayers...")
    gui.root.after(0, lambda: gui.update_progress(97, "Locating Monolayers..."))

    # Find contours on the downscaled post-processed image
    scaled_contours, _ = cv2.findContours(post_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale up the contours to match the original resolution
    contours = [contour * downscale_factor for contour in scaled_contours]

    for i, contour in enumerate(contours):
        # Save the bounding box of the monolayer
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding to the bounding box
        x_start = max(x - config.MONOLAYER_CROP_PADDING, 0)
        y_start = max(y - config.MONOLAYER_CROP_PADDING, 0)
        x_end = min(x + w + config.MONOLAYER_CROP_PADDING, canvas.shape[1])
        y_end = min(y + h + config.MONOLAYER_CROP_PADDING, canvas.shape[0]) 

        # Crop the monolayer from the original canvas
        image_section = canvas[y_start:y_end, x_start:x_end]

        # Create a Monolayer object and add it to the list
        monolayers.append(Monolayer(contour, image_section, (x_start, y_start)))

        cx, cy = monolayers[-1].position
        
        # Mark center of the monolayer
        contour_image = cv2.circle(contour_image, (cx, cy), config.MONOLAYER_DOT_SIZE, color=(0, 0, 0, 255), thickness=-1)
    
    # Draw contours on the original resolution image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 0, 255), config.MONOLAYER_OUTLINE_THICKNESS)
    
    # Display the final image with contours
    gui.root.after(0, lambda: gui.update_progress(97, "Saving final images..."))
    print("Saving image with monolayers...")
    save_image(contour_image, "Images/highlighted_monolayers.png", 5)
    print("Saved!")

    # Sort monolayers by area removing any which are too small
    monolayers = [layer for layer in monolayers if layer.area_um >= downscale_factor]
    monolayers.sort(key=operator.attrgetter('area'))
    monolayers.reverse()

    gui.root.after(0, lambda: gui.display_results_tab(Image.fromarray(cv2.cvtColor(scale_down_canvas(contour_image, config.RESULTS_IMAGE_DOWNSCALE), cv2.COLOR_BGRA2RGBA)), monolayers))


    for i, layer in enumerate(monolayers):
        print(f"{i+1}: Area: {layer.area_um:.0f} um^2,  Centre: {layer.position}      Entropy: {layer.smoothed_entropy:.2f}, TV Norm: {layer.total_variation_norm:.2f}, Local Intensity Variance: {layer.local_intensity_variance:.2f}, CNR: {layer.contrast_to_noise_ratio:.2f}, Skewness: {layer.skewness:.2f}")
        cv2.imwrite(f"Monolayers/{i+1}.png", layer.image)


    gui.root.after(0, lambda: gui.update_progress(100, "Scan Complete!"))
    # while True:
    #     try:
    #         n = int(input("Go To Monolayer: "))-1
    #     except ValueError:
    #         print("Please enter a valid monolayer number")
    #     if n in range(0, len(monolayers)):
    #         move(mcm301obj, image_to_stage(monolayers[n].position, start))
    #     else:
    #         print("Please enter a valid monolayer number")


def alg(gui, mcm301obj, image_queue, frame_queue, start, end):
    """
    Runs the main scanning algorithm to capture and process images across a defined area.
    
    Args:
        mcm301obj (MCM301): The stage controller object.
        image_queue (queue.Queue): Queue containing images captured by the camera.
        frame_queue (queue.Queue): Queue to store images along with their positions for stitching.
        start (tuple): Starting coordinates (x, y) in nanometers.
        end (tuple): Ending coordinates (x, y) in nanometers.
    """

    num_images = math.ceil((end[0]-start[0])/config.DIST+1)*math.ceil((end[1]-start[1])/config.DIST+1)
    print(num_images)
    config.current_image = 0

    focuses = []
    
    def capture_and_store_frame(x, y, focuses):
        """
        Captures a frame from the image queue and stores it with its position in the frame queue.
        
        Args:
            x (int): The x-coordinate in nanometers.
            y (int): The y-coordinate in nanometers.
        """
        time.sleep(1.5*config.CAMERA_PROPERTIES["exposure"]/1e6)


        ################################################################################################################################### Change this back
        frame = image_queue.get(timeout=1000)

        if config.AUTOFOCUS:
            focuses.append(get_focus(frame))
            if len(focuses) > config.FOCUS_FRAME_AVG*2 and np.average(focuses[0:-config.FOCUS_FRAME_AVG]) > np.average(focuses[-config.FOCUS_FRAME_AVG:-1])*config.FOCUS_BUFFER:
                auto_focus(mcm301obj, image_queue)
                focuses.clear()
                frame = image_queue.get(timeout=1000)
                focuses.append(get_focus(frame))


        r = random.randint(-4, 4)
        if r > 0:
            frame = Image.open(f"Images/test_image{r}.jpg")
        frame_queue.put((frame, (x, y)))
        config.current_image += 1
        gui.root.after(0, lambda: gui.update_progress(int(90*config.current_image/num_images), f"Capturing Image {config.current_image} of {num_images}"))

        ###################################################################################################################################

    def scan_line(x, y, direction, focuses):
        """
        Scans a single line in the current direction, capturing images along the way.
        
        Args:
            x (int): The current x-coordinate in nanometers.
            y (int): The current y-coordinate in nanometers.
            direction (int): The scanning direction (1 for forward, -1 for backward).
            
        Returns:
            int: The updated x-coordinate after completing the line scan.
        """
        while (direction == 1 and x < end[0]) or (direction == -1 and x > start[0]):
            capture_and_store_frame(x, y, focuses)
            x += config.DIST * direction
            move(mcm301obj, (x, y))
        
        # Capture final frame at the line end
        capture_and_store_frame(x, y, focuses)
        
        return x
    
    # Start scanning
    move(mcm301obj, start)
    x, y = start
    direction = 1

    gui.root.after(0, lambda: gui.auto_focus_button.config(state='disabled'))
    if config.AUTOFOCUS:
        gui.root.after(0, lambda: gui.update_progress(1, "Focusing..."))
        auto_focus(mcm301obj, image_queue, [config.FOCUS_RANGE*2, config.FOCUS_STEPS*2])
    
    while y < end[1]:
        x = scan_line(x, y, direction, focuses)
        
        # Move to the next line
        y += config.DIST
        move(mcm301obj, (x, y))
        
        # Reverse direction for the next line scan
        direction *= -1
    scan_line(x, y, direction, focuses)

    print("\nImage capture complete!")
    print("Waiting for image processing to complete...")
    gui.root.after(0, lambda: gui.update_progress(92, "Image stitching..."))
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='normal'))
    frame_queue.put(None)


def auto_focus(mcm301obj, image_queue, params=None):
    z = get_pos(mcm301obj, (6,))[0]
    lens_ajustment_factor = 20/config.CAMERA_PROPERTIES['lens']
    if params:
        z_range = np.linspace(z-params[0], z+params[0], params[1]) 
    else:
        z_range = np.linspace(z-int(config.FOCUS_RANGE*lens_ajustment_factor), z+int(config.FOCUS_RANGE*lens_ajustment_factor), config.FOCUS_STEPS)
    best_z = z
    best_focus = 0
    for z_i in z_range:
        move(mcm301obj, [int(z_i)], (6,))
        time.sleep(config.CAMERA_PROPERTIES['exposure']/1e6)
        focus = get_focus(image_queue.get(1000))
        if focus > best_focus:
            best_focus = focus
            best_z = int(z_i)
    move(mcm301obj, [best_z], (6,))

def initial_auto_focus(gui, mcm301obj, image_queue, n = 5):
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='disabled'))
    d = 1e7/config.CAMERA_PROPERTIES['lens']
    while d > 10e4/config.CAMERA_PROPERTIES['lens']:
        auto_focus(mcm301obj, image_queue, params=[int(d), n])
        d = d/(n-1)
    gui.root.after(0, lambda: gui.auto_focus_button.config(state='normal'))

def get_focus(image):
    """
    Calculates the focus measure of an image using the Power Spectrum Slope method.

    Parameters:
        image (PIL.Image): Input PIL image.

    Returns:
        float: Focus measure value (slope of the power spectrum).
    """
    # Convert to grayscale and convert to numpy array
    img_gray = np.array(image.convert('L'), dtype=np.float32)

    # Apply a Hamming window to reduce edge effects
    window = hamming(img_gray.shape[0])[:, None] * hamming(img_gray.shape[1])[None, :]
    img_windowed = img_gray * window

    # Compute the 2D FFT of the image
    f = np.fft.fft2(img_windowed)
    fshift = np.fft.fftshift(f)

    # Compute the power spectrum
    power_spectrum = np.abs(fshift) ** 2

    # Create a grid of frequencies
    num_rows, num_cols = img_gray.shape
    cy, cx = num_rows // 2, num_cols // 2
    y, x = np.indices((num_rows, num_cols))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(int)  # Changed from np.int to int

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
        slope, intercept = curve_fit(linear_func, log_freq, log_power)[0]
    except RuntimeError:
        # If the fit fails, return a large negative slope indicating poor focus
        slope = -10.0

    # The slope of the line is the focus measure
    focus_value = -slope  # Invert the slope to make higher values indicate better focus

    print(f"Focus Value: {focus_value}")

    return focus_value









class Monolayer:
    def __init__(self, contour, image, pos):
        self.image = image
        self.global_contour = contour

        x_start, y_start = pos

        M = cv2.moments(contour)
        if M['m00'] != 0:
            self.cx = int(M['m10']/M['m00'])
            self.cy = int(M['m01']/M['m00'])
        else:
            # Edge case: Area is zero; calculate centroid from the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            self.cx = x + w // 2 
            self.cy = y + h // 2 
        self.area_px = cv2.contourArea(contour)
        self.position = (self.cx, self.cy)
        self.area = self.area_px * (config.NM_PER_PX**2)
        self.area_um = self.area / 1e6

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

        # Get the canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

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

        # Move the stage accordingly
        move_relative(self.stage_controller, [stage_move_x, stage_move_y], wait=False)



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

        # Corner 1 Button
        corner1_button = tk.Button(
            scan_area_frame,
            text="Corner 1",
            command=self.update_corner1_position,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        corner1_button.pack(side=tk.LEFT, padx=10)

        # Corner 2 Button
        corner2_button = tk.Button(
            scan_area_frame,
            text="Corner 2",
            command=self.update_corner2_position,
            bg=config.BUTTON_COLOR,
            fg=config.TEXT_COLOR,
            font=config.BUTTON_FONT
        )
        corner2_button.pack(side=tk.LEFT, padx=10)

        # Labels to display the positions of Corner 1 and Corner 2
        self.corner1_position_label = tk.Label(
            self.main_frame_controls,
            text="Corner 1: Not Set",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        self.corner1_position_label.grid(row=1, column=0, pady=5, sticky='w', columnspan=2)

        self.corner2_position_label = tk.Label(
            self.main_frame_controls,
            text="Corner 2: Not Set",
            bg=config.THEME_COLOR,
            fg=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        self.corner2_position_label.grid(row=2, column=0, pady=5, sticky='w', columnspan=2)

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
        self.corner1_position_label.config(text=f"Corner 1: {formatted_position}")
        self.update_start_end(config.start_pos, config.end_pos)

    def update_corner2_position(self):
        """Update Corner 2 position label."""
        position = get_pos(self.stage_controller, (4, 5))
        config.corner2.__setitem__(slice(None), position)
        formatted_position = [f"{p:.2e}" for p in position]
        self.corner2_position_label.config(text=f"Corner 2: {formatted_position}")
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
                    args=(self, config.canvas, config.start_pos),
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
        self.create_calibration_controls()
        self.create_lens_selector()
        self.create_auto_focus()

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

    def create_calibration_controls(self):
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

        # Navigation buttons positions
        navigation_buttons = [
            ("Up", (0, 1), lambda: move_relative(self.stage_controller, pos=[int(-config.DIST)], stages=(5,), wait=False)),
            ("Left", (1, 0), lambda: move_relative(self.stage_controller, pos=[int(-config.DIST)], stages=(4,), wait=False)),
            ("Right", (1, 2), lambda: move_relative(self.stage_controller, pos=[int(config.DIST)], stages=(4,), wait=False)),
            ("Down", (2, 1), lambda: move_relative(self.stage_controller, pos=[int(config.DIST)], stages=(5,), wait=False)),
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
            ("Focus +", 0, lambda: move_relative(self.stage_controller, pos=[10000], stages=(6,), wait=False)),
            ("Focus -", 1, lambda: move_relative(self.stage_controller, pos=[-10000], stages=(6,), wait=False)),
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
            config.NM_PER_PX = config.CAMERA_PROPERTIES['px_size']*1000/lens_value
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
            target = (config.BRIGHTNESS_RANGE[0] + config.BRIGHTNESS_RANGE[1]) / 2
            self.camera.gain = int(test_gain)
            self.camera.exposure_time_us = int(test_exposure)
            time.sleep(2*test_exposure/1e6)
            for i in range(3):
                latest_frame = np.array(self.image_acquisition_thread._image_queue.get(max(int(test_exposure/1e3), 1000)))
            average_intensity = latest_frame.mean()
            
            while not (config.BRIGHTNESS_RANGE[0] <= average_intensity <= config.BRIGHTNESS_RANGE[1]):
                delta_intensity = target - average_intensity
                
                if 1 < test_gain < 350:
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
                    test_gain = min(max(1, test_gain), 350)
                else:
                    if average_intensity > 240:
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
                print(f"Gain: {test_gain}, Exposure: {test_exposure}")

            # Update config with final gain and exposure values
            config.CAMERA_PROPERTIES['gain'] = int(test_gain)
            config.CAMERA_PROPERTIES['exposure'] = int(test_exposure)

            # Update GUI entries
            self.gain_entry.delete(0, tk.END)
            self.gain_entry.insert(0, str(config.CAMERA_PROPERTIES['gain'] / 10))
            self.exposure_entry.delete(0, tk.END)
            self.exposure_entry.insert(0, str(config.CAMERA_PROPERTIES['exposure'] / 1000))

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

    x_1, y_1 = start
    x_2, y_2 = end
    start = [min(x_1, x_2), min(y_1, y_2)]
    end = [max(x_1, x_2), max(y_1, y_2)]
    
    # Disable buttons in the main tab
    gui.root.after(0, lambda: gui.toggle_buttons(gui.main_frame_controls ,'disabled', exclude=[gui.cat_button]))
    gui.root.after(0, lambda: gui.display_results_tab(Image.open("assets/placeholder.webp")))
    gui.image_references = []

    # Start stitching thread
    stitching_thread = threading.Thread(target=stitch_and_display_images, args=(gui, frame_queue, start, end, stitched_view_canvas))
    stitching_thread.start()

    # Start algorithm thread
    alg_thread = threading.Thread(target=alg, args=(gui, mcm301obj, image_queue, frame_queue, start, end))
    alg_thread.start()

    # Define a function to check if stitching is complete
    def check_stitching_complete():
        stitching_thread.join()  # Wait for stitching to finish
        gui.root.after(0, lambda: gui.toggle_buttons(gui.main_frame_controls ,'normal'))  # Enable buttons again

    # Start a separate thread to re-enable buttons once stitching is done
    threading.Thread(target=check_stitching_complete, daemon=True).start()

def cleanup(signum=None, frame=None):
    logging.info("Signal received, cleaning up...")
    try:
        if 'GUI_main' in globals() and GUI_main is not None:
            GUI_main.on_closing()
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
    sys.exit(0)



if __name__ == "__main__":
    # Set up signal handlers for unexpected termination
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all types of logs
        format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler('app_debug.log', mode='w')  # Log to a file
        ]
    )

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
            logging.info(f"Camera {camera_list[0]} initialized.")

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

            # Proceed with your application logic
            try:
                # Update config parameters
                config.CAMERA_PROPERTIES["px_size"] = (
                    camera.sensor_pixel_height_um + camera.sensor_pixel_width_um
                ) / 2
                config.NM_PER_PX = (
                    config.CAMERA_PROPERTIES["px_size"] * 1000 / config.CAMERA_PROPERTIES["lens"]
                )

                config.CAMERA_DIMS = [camera.image_width_pixels, camera.image_height_pixels]
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
                if GUI_main.image_acquisition_thread.is_alive():
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
