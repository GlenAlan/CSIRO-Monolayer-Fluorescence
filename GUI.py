import os
import sys
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
        self.is_active = True  # Initialize attribute to control image updates
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


    # print("Saving post processed image...")
    # save_image(post_image, "Images/processed.png")
    # print("Saved!")

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
    save_image(contour_image, "Images/highlighted_monolayers.png", 5)
    print("Saved!")

    # Sort monolayers by area removing any which are too small
    monolayers = [layer for layer in monolayers if layer.area_um >= downscale_factor]
    monolayers.sort(key=operator.attrgetter('area'))
    monolayers.reverse()

    gui.display_results_tab(Image.fromarray(cv2.cvtColor(scale_down_canvas(contour_image, 10), cv2.COLOR_BGRA2RGBA)), monolayers)


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
    gui.update_progress(92, "Image stitching...")
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
        # Initialize the main window
        self.root = root
        self.root.title('Camera Control Interface')
        self.root.configure(bg=config.THEME_COLOR)

        # Initialize camera and image acquisition thread
        self.camera = camera
        self.image_acquisition_thread = ImageAcquisitionThread(self.camera, rotation_angle=270)
        self.image_acquisition_thread.start()

        self.frame_queue = queue.Queue()

        self.image_references = []

        # Initialize the stage controller
        self.stage_controller = stage_setup(home=False)

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

        # Initialize live position labels in the left frame
        self.position_frame = tk.Frame(self.left_frame, bg=config.THEME_COLOR)
        self.position_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.init_position_labels()

        # Start live position updates
        self.update_positions()

        # Create control buttons and other UI elements
        self.create_control_buttons()
        self.create_rotation_wheel()

        # Initialize progress bar and status label
        self.init_progress_bar()
        self.init_main_buttons()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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
        self.stitched_view_canvas = LiveViewCanvas(self.tabs["main"], queue.Queue())
        self.stitched_view_canvas.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Configure grid in 'main' tab
        self.tabs["main"].grid_columnconfigure(0, weight=1)
        self.tabs["main"].grid_columnconfigure(1, weight=1)
        self.tabs["main"].grid_rowconfigure(0, weight=1)

        # Calibration Tab Layout
        self.calibration_frame_controls = tk.Frame(self.tabs["calibration"], bg=config.THEME_COLOR)
        self.calibration_frame_controls.pack(fill="both", expand=True, padx=10, pady=10)

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

        def update_image_on_resize(event=None):
            canvas_width = max(self.results_frame_image.winfo_width(), MIN_WIDTH)
            canvas_height = max(self.results_frame_image.winfo_height(), MIN_HEIGHT)

            if canvas_width <= 0 or canvas_height <= 0:
                return

            if image.width == 0 or image.height == 0:
                print("Invalid image dimensions.")
                return

            aspect_ratio = image.width / image.height

            if canvas_width / aspect_ratio < canvas_height:
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)

            new_width = max(1, int(new_width))
            new_height = max(1, int(new_height))

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            self._resized_image = ImageTk.PhotoImage(resized_image)
            self.results_frame_image.delete("all")
            self.results_frame_image.create_image(0, 0, anchor='nw', image=self._resized_image)

        # Set minimum size and disable shrinking below this size
        self.results_frame_image.config(width=MIN_WIDTH, height=MIN_HEIGHT)
        self.results_frame_image.grid_propagate(False)  # Prevent the canvas from shrinking below its minimum size

        # Bind event to dynamically update image size when the canvas is resized
        self.results_frame_image.bind("<Configure>", update_image_on_resize)

        # Delay the initial call to allow GUI to initialize
        self.root.after(100, update_image_on_resize)

        if monolayers is not None:
            # Clear existing widgets
            for widget in self.results_frame_controls.winfo_children():
                widget.destroy()

            # Define columns
            columns = ('Area (um^2)', 'Centre', 'Entropy', 'TV Norm', 'Intensity Variance', 'CNR', 'Skewness', 'Index')
            visible_rows = 4  # Increased number of visible rows for taller Treeview

            # Create a frame to hold the Treeview and scrollbar
            tree_frame = tk.Frame(self.results_frame_controls, bg=config.THEME_COLOR)
            tree_frame.grid(row=0, column=0, sticky='nsew')

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


    def stop_image_updates(self):
        """Stop image updates when not on the main or calibration tabs."""
        self.main_frame_image.set_active(False)
        self.calibration_frame_image.set_active(False)

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
            ("Up", (0, 1), lambda: move_relative(self.stage_controller, pos=[int(-config.DIST / 2)], stages=(5,), wait=False)),
            ("Left", (1, 0), lambda: move_relative(self.stage_controller, pos=[int(-config.DIST / 2)], stages=(4,), wait=False)),
            ("Right", (1, 2), lambda: move_relative(self.stage_controller, pos=[int(config.DIST / 2)], stages=(4,), wait=False)),
            ("Down", (2, 1), lambda: move_relative(self.stage_controller, pos=[int(config.DIST / 2)], stages=(5,), wait=False)),
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


    def create_rotation_wheel(self):
        """Create a 360-degree rotation wheel in the calibration tab."""
        wheel_frame = tk.Frame(self.calibration_frame_controls, bg=config.THEME_COLOR)
        wheel_frame.grid(row=11, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

        rotation_wheel_label = tk.Label(
            self.calibration_frame_controls,
            text="360 Degree Wheel:",
            background=config.THEME_COLOR,
            foreground=config.TEXT_COLOR,
            font=config.LABEL_FONT
        )
        rotation_wheel_label.grid(row=12, column=0, pady=10, columnspan=2)

        self.canvas = tk.Canvas(wheel_frame, width=200, height=200, bg=config.THEME_COLOR, highlightthickness=0)
        self.canvas.pack()

        self.wheel_radius = 80
        self.wheel_center_x = 100
        self.wheel_center_y = 100

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

        self.angle_entry = tk.Entry(wheel_frame, width=5, font=config.LABEL_FONT)
        self.angle_entry.pack(pady=10)
        self.angle_entry.insert(0, "270")
        self.angle_entry.bind("<Return>", self.set_angle_from_entry)
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
        try:
            if self.image_acquisition_thread.is_alive():
                self.image_acquisition_thread.stop()
                self.image_acquisition_thread.join()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        self.root.destroy()



def run_sequence(gui, mcm301obj, image_queue, frame_queue, start, end, stitched_view_canvas):

    x_1, y_1 = start
    x_2, y_2 = end
    start = [min(x_1, x_2), min(y_1, y_2)]
    end = [max(x_1, x_2), max(y_1, y_2)]
    
    # Disable buttons in the main tab
    gui.toggle_buttons(gui.main_frame_controls ,'disabled', exclude=[gui.cat_button])
    gui.display_results_tab(Image.open("assets/placeholder.webp"))
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
        gui.toggle_buttons(gui.main_frame_controls ,'normal')  # Enable buttons again

    # Start a separate thread to re-enable buttons once stitching is done
    threading.Thread(target=check_stitching_complete, daemon=True).start()



if __name__ == "__main__":
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        # print(camera_list)
        if len(camera_list) == 0:
            print("No cameras found.")
            sys.exit(1)

        with sdk.open_camera(camera_list[-1]) as camera:
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
