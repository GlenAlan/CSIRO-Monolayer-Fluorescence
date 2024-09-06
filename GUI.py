import os
import sys
import time
import random
import math
import typing
import threading
import queue
import operator
from concurrent.futures import ThreadPoolExecutor

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw

import cv2  # OpenCV library for accessing the webcam
import numpy as np
from skimage.measure import shannon_entropy
import torch

from custom_definitions import *
import config

from MCM301_COMMAND_LIB import *

# Thorlabs TSI SDK imports
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

# Windows-specific setup
try:
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None


def image_to_stage(center_coords, start):
    center_coords_raw = (int((center_coords[0] - config.CAMERA_DIMS[0])* config.NM_PER_PX + start[0]), int((center_coords[1] - config.CAMERA_DIMS[1])* config.NM_PER_PX + start[1]))
    return center_coords_raw


class LiveViewCanvas(tk.Canvas):
    def __init__(self, parent, image_queue):
        super().__init__(parent, bg=config.THEME_COLOR, highlightthickness=0)
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0
        self.is_active = False  # Initialize attribute to control image updates
        self.bind("<Configure>", self.on_resize)  # Bind resizing event
        self._display_image()

    def _display_image(self):
        if not self.is_active:
            self.after(config.UPDATE_DELAY, self._display_image)
            return

        try:
            image = self.image_queue.get_nowait()
            
            # Resize image to match canvas size while maintaining aspect ratio
            self.update_image_display(image)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating image: {e}")

        self.after(config.UPDATE_DELAY, self._display_image)  # Continue updating the image

    def update_image_display(self, image):
        """Resize image based on canvas size while maintaining aspect ratio."""
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        aspect = image.size[0] / image.size[1]
        
        if canvas_width / aspect < canvas_height:
            new_width = canvas_width
            new_height = int(canvas_width / aspect)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * aspect)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        self._image = ImageTk.PhotoImage(master=self, image=image)
        self.create_image(0, 0, image=self._image, anchor='nw')

    def on_resize(self, event):
        """Handle canvas resizing."""
        self._image_width = event.width
        self._image_height = event.height
        self._display_image()  # Update display to fit new canvas size

    def set_active(self, active):
        """Enable or disable image updating based on tab visibility."""
        self.is_active = active


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

            scaled_canvas = scale_down_canvas(canvas, 10)

            stitched_image = Image.fromarray(cv2.cvtColor(scaled_canvas, cv2.COLOR_BGRA2RGBA))
            stitched_view_canvas.update_image_display(stitched_image)

        # Wait for all futures to complete
        for future in futures:
            future.result()

    # End timing after all processing is complete
    t2 = time.time()
    print("Image stitching complete")
    gui.update_progress(92, "Image stitching...")
    print(f"Time taken: {t2 - t1:.2f} seconds")

    print("Saving final image...")
    #cv2.imwrite("Images/final.png", canvas)
    #cv2.imwrite("Images/final_compressed.png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    save_image(canvas, "Images/final_downsampled.png", 10)

    print("Save complete")

    scaled_canvas = scale_down_canvas(canvas, 10)

    stitched_image = Image.fromarray(cv2.cvtColor(scaled_canvas, cv2.COLOR_BGRA2RGBA))
    stitched_view_canvas.update_image_display(stitched_image)

    post_processing(gui, canvas, start)

def post_processing(gui, canvas, start, contrast=2, threshold=100):
    t1 = time.time()
    gui.update_progress(95, "Image processing...")
    print("Post processing...")

    downscale_factor = 4
    monolayers = []

    # Create a downsampled version of the canvas for post-processing
    post_image = scale_down_canvas(canvas, downscale_factor)

    # post_image = cv2.blur(post_image, (5, 5)) # Optional pre grayscale blur (downscaling already blurs)
    # Our image is in BGRA format so to convert it to greyscale with a bias for red and a bias against green and blue, we use the following formula:
    post_image = 1 * post_image[:, :, 2] - 1.0 * post_image[:, :, 1] - 0.0 * post_image[:, :, 0]
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


    print("Saving post processed image...")
    save_image(post_image, "Images/processed.png")
    print("Saved!")

    # Create a copy of the canvas for contour drawing (this is slow but necessary if canvas is to be used again in future)
    contour_image = canvas.copy()

    print("Locating Monolayers...")
    gui.update_progress(97, "Locating Monolayers...")

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
        contour_image = cv2.circle(contour_image, (cx, cy), 5, color=(0, 0, 0, 255), thickness=-1)
    
    # Draw contours on the original resolution image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 0, 255), 5)
    
    # Display the final image with contours
    gui.update_progress(97, "Saving final images...")
    print("Saving image with monolayers...")
    save_image(contour_image, "Images/contour.png", 5)
    print("Saved!")

    # Sort monolayers by area removing any which are too small
    monolayers = [layer for layer in monolayers if layer.area_um >= downscale_factor]
    monolayers.sort(key=operator.attrgetter('area'))

    for i, layer in enumerate(monolayers):
        print(f"{i+1}: Area: {layer.area_um:.0f} um^2,  Centre: {layer.position}      Entropy: {layer.smoothed_entropy:.2f}, TV Norm: {layer.total_variation_norm:.2f}, Local Intensity Variance: {layer.local_intensity_variance:.2f}, CNR: {layer.contrast_to_noise_ratio:.2f}, Skewness: {layer.skewness:.2f}")
        cv2.imwrite(f"Monolayers/{i+1}.png", layer.image)


    gui.update_progress(100, "Scan Complete!")
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
    
    def capture_and_store_frame(x, y):
        """
        Captures a frame from the image queue and stores it with its position in the frame queue.
        
        Args:
            x (int): The x-coordinate in nanometers.
            y (int): The y-coordinate in nanometers.
        """
        time.sleep(config.CAMERA_PROPERTIES["exposure"]/1e6)


        ################################################################################################################################### Change this back
        frame = image_queue.get(timeout=1000)
        r = random.randint(-4, 4)
        if r > 0:
            frame = Image.open(f"Images/test_image{r}.jpg")
        frame_queue.put((frame, (x, y)))
        config.current_image += 1
        gui.update_progress(int(90*config.current_image/num_images), f"Capturing Image {config.current_image} of {num_images}")

        ###################################################################################################################################

    def scan_line(x, y, direction):
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
            capture_and_store_frame(x, y)
            x += config.DIST * direction
            move(mcm301obj, (x, y))
        
        # Capture final frame at the line end
        capture_and_store_frame(x, y)
        
        return x
    
    # Start scanning
    move(mcm301obj, start)
    x, y = start
    direction = 1
    
    while y < end[1]:
        x = scan_line(x, y, direction)
        
        # Move to the next line
        y += config.DIST
        move(mcm301obj, (x, y))
        
        # Reverse direction for the next line scan
        direction *= -1
    scan_line(x, y, direction)

    print("\nImage capture complete!")
    print("Waiting for image processing to complete...")
    frame_queue.put(None)

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
        self.root = root
        self.root.title('Camera Control Interface')
        self.root.configure(bg=config.THEME_COLOR)

        self.camera = camera
        self.image_acquisition_thread = ImageAcquisitionThread(self.camera, rotation_angle=270)
        self.image_acquisition_thread.start()
        self.vel = 50  # Not sure what this does         
    
        self.frame_queue = queue.Queue()

        # Initialize MCM301 object
        self.mcm301obj = stage_setup()

        # Notebook and Tabs Setup (initialize self.tabs here)
        self.setup_notebook()

        # Now that self.tabs is initialized, we can safely create the stitched view canvas
        self.stitched_view_canvas = LiveViewCanvas(self.tabs["main"], queue.Queue())
        self.stitched_view_canvas.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        # Flags to control image updates
        self.update_active_main = False
        self.update_active_calib = False

        # Create UI Elements
        self.create_control_buttons()
        self.create_sliders()
        self.create_360_wheel()

        # Initialize Live Position Labels
        self.init_position_labels()

        # Start Live Position Updates
        self.update_positions()

        # Initialize Progress Bar and Status Label
        self.init_progress_bar()
        self.init_main_buttons()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_progress_bar(self):
        """Initialize the progress bar and status label in the main tab."""
        self.progress_value = tk.DoubleVar()
        self.progress_status = tk.StringVar()
        self.progress_status.set("Idle...")

        # Progress bar widget
        self.progress_bar = ttk.Progressbar(self.main_frame_text, orient="horizontal", length=200, mode="determinate", variable=self.progress_value)
        self.progress_bar.grid(row=10, column=0, pady=10, sticky='ew', columnspan=2)

        # Progress status label
        self.progress_label = tk.Label(self.main_frame_text, textvariable=self.progress_status, bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
        self.progress_label.grid(row=11, column=0, pady=5, sticky='w', columnspan=2)

    def init_main_buttons(self):
        """Initialize the buttons in the main tab."""
        # Frame for the buttons
        scan_area_frame = tk.Frame(self.main_frame_text, bg=config.THEME_COLOR)
        scan_area_frame.grid(row=5, column=0, pady=10, sticky='w', columnspan=2)

        # Corner 1 Button
        start_button = tk.Button(scan_area_frame, text="Corner 1", 
                                command=self.update_corner1_position,
                                bg=config.BUTTON_COLOR, fg=config.TEXT_COLOR, font=config.BUTTON_FONT)
        start_button.pack(side=tk.LEFT, padx=10)

        # Corner 2 Button
        end_button = tk.Button(scan_area_frame, text="Corner 2", 
                            command=self.update_corner2_position,
                            bg=config.BUTTON_COLOR, fg=config.TEXT_COLOR, font=config.BUTTON_FONT)
        end_button.pack(side=tk.LEFT, padx=10)


        # Labels to display the positions of Corner 1 and Corner 2
        self.corner1_pos_label = tk.Label(self.main_frame_text, text="Corner 1: Not Set", 
                                        bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
        self.corner1_pos_label.grid(row=6, column=0, pady=5, sticky='w', columnspan=2)

        self.corner2_pos_label = tk.Label(self.main_frame_text, text="Corner 2: Not Set", 
                                        bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
        self.corner2_pos_label.grid(row=7, column=0, pady=5, sticky='w', columnspan=2)

        alg_button = tk.Button(scan_area_frame, text="Begin Search", 
           command=lambda: run_sequence(self, self.mcm301obj, self.image_acquisition_thread._image_queue, 
                               self.frame_queue, config.start_pos, config.end_pos, self.stitched_view_canvas),
           bg=config.BUTTON_COLOR, fg=config.TEXT_COLOR, font=config.BUTTON_FONT)
        alg_button.pack(side=tk.BOTTOM, padx=10, pady=10)

    def update_corner1_position(self):
        """Update Corner 1 position label."""
        pos = get_pos(self.mcm301obj, (4, 5))
        config.start_pos.__setitem__(slice(None), pos)
        formatted_pos = [f"{p:.2e}" for p in pos]
        self.corner1_pos_label.config(text=f"Corner 1: {formatted_pos}")

    def update_corner2_position(self):
        """Update Corner 2 position label."""
        pos = get_pos(self.mcm301obj, (4, 5))
        config.end_pos.__setitem__(slice(None), pos)
        formatted_pos = [f"{p:.2e}" for p in pos]
        self.corner2_pos_label.config(text=f"Corner 2: {formatted_pos}")

    def toggle_buttons(self, widget, state: str):
        """
        Enable or disable all buttons in the main tab, including nested children.

        Args:
            state (str): The state of the buttons ('normal' to enable, 'disabled' to disable).
        """
        def recursive_toggle(widget):
            """Recursively toggle buttons in the widget and its children."""
            for child in widget.winfo_children():
                if isinstance(child, tk.Button):
                    child.config(state=state)
                # Recursively call this function on any children widgets (for nested frames, etc.)
                recursive_toggle(child)

        # Start the recursive toggle with the main frame
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
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")

        self.tabs = {
            "main": ttk.Frame(notebook, style='TFrame'),
            "calibration": ttk.Frame(notebook, style='TFrame'),
            "results": ttk.Frame(notebook, style='TFrame'),
            "devices": ttk.Frame(notebook, style='TFrame')
        }

        notebook.add(self.tabs["main"], text="Main Control")
        notebook.add(self.tabs["calibration"], text="Calibration")
        notebook.add(self.tabs["results"], text="Results & Analysis")
        notebook.add(self.tabs["devices"], text="Devices")

        # Apply styles
        style = ttk.Style()
        style.theme_use('default')

        # Configure styles for the notebook
        style.configure('TFrame', background=config.THEME_COLOR)
        style.configure('TNotebook', background=config.THEME_COLOR, foreground=config.TEXT_COLOR)
        style.configure('TNotebook.Tab', 
                        background=config.BUTTON_COLOR, 
                        foreground=config.TEXT_COLOR, 
                        font=config.LABEL_FONT, 
                        padding=(5, 1))  # Add padding for better aesthetics
        style.map('TNotebook.Tab', 
                background=[('selected', config.HIGHLIGHT_COLOR)], 
                foreground=[('selected', config.TEXT_COLOR)],
                expand=[('selected', [1, 1, 1, 0])])  # Makes the selected tab standout
        style.configure("TProgressbar", troughcolor=config.THEME_COLOR, background=config.HIGHLIGHT_COLOR, thickness=20)

        # Bind tab change event
        notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Main Control Tab Layout
        self.main_frame_image = LiveViewCanvas(self.tabs["main"], self.image_acquisition_thread._image_queue)
        self.main_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        self.main_frame_text = tk.Frame(self.tabs["main"], bg=config.THEME_COLOR)
        self.main_frame_text.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Calibration Tab Layout
        self.calib_frame_image = LiveViewCanvas(self.tabs["calibration"], self.image_acquisition_thread._image_queue)
        self.calib_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        self.calib_frame_text = tk.Frame(self.tabs["calibration"], bg=config.THEME_COLOR)
        self.calib_frame_text.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # Responsive Layout Configuration
        for tab in self.tabs.values():
            tab.grid_columnconfigure(0, weight=3)
            tab.grid_columnconfigure(1, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        notebook.grid_columnconfigure(0, weight=1)
        notebook.grid_rowconfigure(0, weight=1)

        self.main_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.calib_frame_image.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')


    def init_position_labels(self):
        """Initialize position labels for live updates."""
        self.pos_names = ["Pos X", "Pos Y", "Focus Z"]
        self.position_labels_main = []
        self.position_labels_calib = []

        for i, name in enumerate(self.pos_names):
            label_main = tk.Label(self.main_frame_text, text=name, padx=10, pady=5, bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
            label_main.grid(row=i, column=0, sticky='w')
            pos_label_main = tk.Label(self.main_frame_text, text="0.00 nm", padx=5, bg=config.BUTTON_COLOR, fg=config.TEXT_COLOR, width=15, font=config.LABEL_FONT)
            pos_label_main.grid(row=i, column=1, sticky='w')
            self.position_labels_main.append(pos_label_main)

            label_calib = tk.Label(self.calib_frame_text, text=name, padx=10, pady=5, bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
            label_calib.grid(row=i, column=0, sticky='w', padx=(10, 5), pady=(5, 5))
            pos_label_calib = tk.Label(self.calib_frame_text, text="0.00 nm", padx=5, bg=config.BUTTON_COLOR, fg=config.TEXT_COLOR, width=15, font=config.LABEL_FONT)
            pos_label_calib.grid(row=i, column=1, sticky='w', padx=(5, 10), pady=(5, 5))
            self.position_labels_calib.append(pos_label_calib)

    def update_positions(self):
        """Update the positions displayed in the GUI."""
        positions = get_pos(self.mcm301obj, stages=(4, 5, 6))
        for i, pos_label in enumerate(self.position_labels_main):
            pos_label.config(text=f"{positions[i]:.2e} nm")

        for i, pos_label in enumerate(self.position_labels_calib):
            pos_label.config(text=f"{positions[i]:.2e} nm")

        # Schedule the next update
        self.root.after(config.POSITION_UPDATE_INTERVAL, self.update_positions)

    def on_tab_change(self, event):
        """Handle tab change events to activate/deactivate live view."""
        selected_tab = event.widget.index("current")
        if selected_tab == 0:  # Main Tab
            self.update_active_main = True
            self.update_active_calib = False
            self.main_frame_image.set_active(True)
            self.calib_frame_image.set_active(False)
        elif selected_tab == 1:  # Calibration Tab
            self.update_active_main = False
            self.update_active_calib = True
            self.main_frame_image.set_active(False)
            self.calib_frame_image.set_active(True)
        else:
            self.update_active_main = False
            self.update_active_calib = False
            self.stop_image_updates()

    def stop_image_updates(self):
        """Stop image updates when not on the main or calibration tabs."""
        self.main_frame_image.set_active(False)
        self.calib_frame_image.set_active(False)

    def create_control_buttons(self):
        self.enter_pos = tk.Label(self.main_frame_text, text="Enter Positions (nm):", bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.HEADING_FONT)
        self.enter_pos.grid(row=3, column=0, pady=10, sticky='w')
        self.enter_focus = tk.Label(self.calib_frame_text, text="Focus Control:", bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.HEADING_FONT)
        self.enter_focus.grid(row=5, column=0, pady=10, sticky='w', columnspan=2)

        self.create_position_entries()
        self.create_main_frame_buttons()
        self.create_calibration_controls()

    def create_position_entries(self):
        self.pos_entry_x = tk.Entry(self.main_frame_text, font=config.LABEL_FONT)
        self.pos_entry_x.grid(row=4, column=0, pady=10, sticky='w')
        self.pos_entry_y = tk.Entry(self.main_frame_text, font=config.LABEL_FONT)
        self.pos_entry_y.grid(row=4, column=1, pady=10, sticky='w')
        self.pos_entry_z = tk.Entry(self.calib_frame_text, font=config.LABEL_FONT)
        self.pos_entry_z.grid(row=4, column=1, pady=10, sticky='w')

        self.pos_entry_x.bind('<Return>', lambda event, type="XY": self.submit_entries(type))
        self.pos_entry_y.bind('<Return>', lambda event, type="XY": self.submit_entries(type))
        self.pos_entry_z.bind('<Return>', lambda event, type="Z": self.submit_entries(type))

    def create_main_frame_buttons(self):
        self.main_frame_text_pos = tk.Frame(self.main_frame_text, bg=config.THEME_COLOR, padx=25)
        self.main_frame_text_pos.grid(row=5, sticky='w', pady=30)

    def create_calibration_controls(self):
        calib_button_focus_label = tk.Label(self.calib_frame_text, text="Focus Slider", bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
        calib_button_focus_label.grid(row=6, column=0, pady=10, columnspan=2, sticky='w')

        # Create navigation buttons for calibration tab
        calib_controls = [
            ("Up", (7, 1), lambda: move_relative(self.mcm301obj, pos=[int(config.DIST / 2)], stages=(5,), wait=False)),
            ("Left", (8, 0), lambda: move_relative(self.mcm301obj, pos=[int(-config.DIST / 2)], stages=(4,), wait=False)),
            ("Right", (8, 2), lambda: move_relative(self.mcm301obj, pos=[int(config.DIST / 2)], stages=(4,), wait=False)),
            ("Down", (9, 1), lambda: move_relative(self.mcm301obj, pos=[int(-config.DIST / 2)], stages=(5,), wait=False)),
            ("Zoom in", (7, 3), lambda: move_relative(self.mcm301obj, pos=[10000], stages=(6,), wait=False)),
            ("Zoom out", (9, 3), lambda: move_relative(self.mcm301obj, pos=[-10000], stages=(6,), wait=False))
        ]

        for text, (row, col), cmd in calib_controls:
            button = tk.Button(self.calib_frame_text, text=text, command=cmd, bg=config.BUTTON_COLOR, fg=config.TEXT_COLOR, font=config.BUTTON_FONT)
            button.grid(row=row, column=col, padx=10, pady=5, sticky='nsew')

    def create_sliders(self):
        slider_frame = tk.Frame(self.calib_frame_text, bg=config.THEME_COLOR)
        slider_frame.grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        slider_focus_label = tk.Label(slider_frame, text="Focus Slider:", bg=config.THEME_COLOR, fg=config.TEXT_COLOR, font=config.LABEL_FONT)
        slider_focus_label.grid(row=0, column=0, pady=10, sticky='w')

        slider_focus = tk.Scale(slider_frame, from_=100000, to=1000000, orient='vertical',
                                command=lambda event: self.on_focus_slider_change(event), bg=config.THEME_COLOR, fg=config.TEXT_COLOR)
        slider_focus.grid(row=1, column=0, padx=20, pady=10, sticky='nsew')

    def create_360_wheel(self):
        wheel_frame = tk.Frame(self.calib_frame_text, bg=config.THEME_COLOR)
        wheel_frame.grid(row=11, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

        camera_wheel_label = tk.Label(self.calib_frame_text, text="360 Degree Wheel:", background=config.THEME_COLOR, foreground=config.TEXT_COLOR, font=config.LABEL_FONT)
        camera_wheel_label.grid(row=12, column=0, pady=10, columnspan=2)

        self.canvas = tk.Canvas(wheel_frame, width=200, height=200, bg=config.THEME_COLOR, highlightthickness=0)
        self.canvas.pack()

        self.radius = 80
        self.center_x = 100
        self.center_y = 100

        self.image = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.draw_anti_aliased_wheel()

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.pointer_line = self.canvas.create_line(self.center_x, self.center_y,
                                                    self.center_x + self.radius, self.center_y,
                                                    width=2, fill=config.HIGHLIGHT_COLOR)
        self.canvas.bind("<B1-Motion>", self.update_wheel)

        self.angle_entry = tk.Entry(wheel_frame, width=5, font=config.LABEL_FONT)
        self.angle_entry.pack(pady=10)
        self.angle_entry.insert(0, "270")
        self.angle_entry.bind("<Return>", self.set_angle_from_entry)
        end_x = self.center_x + self.radius * math.cos(math.radians(270))
        end_y = self.center_y + self.radius * math.sin(math.radians(270))
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)


    def on_focus_slider_change(self, event):
        focus_value = int(event)
        print(f"Focus Slider moved to: {focus_value}")


    def draw_anti_aliased_wheel(self):
        self.draw.ellipse(
            [self.center_x - self.radius, self.center_y - self.radius,
             self.center_x + self.radius, self.center_y + self.radius],
            outline=config.HIGHLIGHT_COLOR, width=2
        )

    def update_wheel(self, event):
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        angle = int(math.degrees(math.atan2(dy, dx))) % 360

        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)

        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, f"{angle:.0f}")
        self.image_acquisition_thread._rotation_angle = float(self.angle_entry.get())

    def submit_entries(self, type="XY"):
        if type == "XY":
            enter_x = self.pos_entry_x.get().strip()
            enter_y = self.pos_entry_y.get().strip()
            if self.validate_entries(enter_x, enter_y):
                threading.Thread(target=self.move_and_update_progress, args=([int(enter_x) * 1, int(enter_y) * 1],), daemon=True).start()
        elif type == "Z":
            enter_z = self.pos_entry_z.get().strip()
            if self.validate_entries(enter_z):
                threading.Thread(target=self.move_and_update_progress, args=([int(enter_z) * 1], (6,)), daemon=True).start()



    ############################## TEST ONLY, REPLACE WITH ACTUAL UPDATES ##############################
    def move_and_update_progress(self, pos, stages=(4, 5)):
        """Move the stage to a position and update the progress bar during the operation."""
        self.update_progress(0, "Moving stage...")

        move(self.mcm301obj, pos, stages, wait=False)  # Simulating movement
        self.root.update_idletasks()  # Ensure UI gets updated during the loop

        self.update_progress(100, "Movement complete.")
    #####################################################################################################

    def validate_entries(self, *entries):
        for entry in entries:
            if not entry.isdigit():
                messagebox.showerror("Input Error", "All fields must be integers!")
                return False
        return True

    def set_angle_from_entry(self, event):
        try:
            angle = float(self.angle_entry.get()) % 360
        except ValueError:
            angle = 0

        end_x = self.center_x + self.radius * math.cos(math.radians(angle))
        end_y = self.center_y + self.radius * math.sin(math.radians(angle))
        self.canvas.coords(self.pointer_line, self.center_x, self.center_y, end_x, end_y)
        self.image_acquisition_thread._rotation_angle = float(self.angle_entry.get())

        print(f"Camera rotation set to: {angle:.2f} degrees")

    def on_closing(self):
        """Handle the cleanup on closing the application."""
        try:
            if self.image_acquisition_thread.is_alive():
                self.image_acquisition_thread.stop()
                self.image_acquisition_thread.join()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        self.root.destroy()


def run_sequence(gui, mcm301obj, image_queue, frame_queue, start, end, stitched_view_canvas):

    # Disable buttons in the main tab
    gui.toggle_buttons(gui.main_frame_text ,'disabled')

    # Start stitching thread
    stitching_thread = threading.Thread(target=stitch_and_display_images, args=(gui, frame_queue, start, end, stitched_view_canvas))
    stitching_thread.start()

    # Start algorithm thread
    alg_thread = threading.Thread(target=alg, args=(gui, mcm301obj, image_queue, frame_queue, start, end))
    alg_thread.start()

    # Define a function to check if stitching is complete
    def check_stitching_complete():
        stitching_thread.join()  # Wait for stitching to finish
        gui.toggle_buttons(gui.main_frame_text ,'normal')  # Enable buttons again

    # Start a separate thread to re-enable buttons once stitching is done
    threading.Thread(target=check_stitching_complete, daemon=True).start()



if __name__ == "__main__":
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        if len(camera_list) == 0:
            print("No cameras found.")
            sys.exit(1)

        with sdk.open_camera(camera_list[0]) as camera:
            print(f"Camera {camera_list[0]} initialized.")
            
            # Ensure the camera is properly armed
            print("Setting camera parameters...")
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.arm(2)

            camera.gain = config.CAMERA_PROPERTIES["gain"]
            camera.exposure_time_us = config.CAMERA_PROPERTIES["exposure"]

            
            config.CAMERA_DIMS = [camera.image_width_pixels, camera.image_height_pixels]
            config.DIST = int(min(config.CAMERA_DIMS) * config.NM_PER_PX * (1-config.IMAGE_OVERLAP))

            camera.issue_software_trigger()

            print("App initializing...")
            root = tk.Tk()
            GUI_main = GUI(root, camera)

            print("App starting")
            root.mainloop()

            print("Waiting for image acquisition thread to finish...")
            GUI_main.image_acquisition_thread.stop()
            GUI_main.image_acquisition_thread.join()

            print("Closing resources...")
    print("App terminated. Goodbye!")
