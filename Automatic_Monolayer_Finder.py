try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import *
# from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
# from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

from MCM301_COMMAND_LIB import *
import time
import random

from custom_definitions import *

import tkinter as tk
from PIL import Image, ImageTk
import typing
import threading
import queue
import cv2
import numpy as np
import operator
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import shannon_entropy
import torch
import os

# These bits indicate that the stage is no longer moving.
confirmation_bits = (2147484928, 2147484930, 2147483904)
monolayer_crop_padding = 10

# camera_dims = [2448, 2048] # This is updated dynamically later
camera_properties = {"gain": 255, "exposure": 150000}
nm_per_px = 171.6
image_overlap = 0.05


accel_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    print("Stage setup complete\n")

    return mcm301obj


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
                if bit[0] not in confirmation_bits:
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


def image_to_stage(center_coords, start):
    center_coords_raw = (int((center_coords[0] - camera_dims[0])* nm_per_px + start[0]), int((center_coords[1] - camera_dims[1])* nm_per_px + start[1]))
    return center_coords_raw


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
    canvas_region = torch.tensor(canvas[y_start:y_end, x_start:x_end], dtype=torch.float32, device=accel_device)
    image_region = torch.tensor(cropped_image[y_offset:y_offset + region_height, x_offset:x_offset + region_width], dtype=torch.float32, device=accel_device)

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
        int((center_coords_raw[0] - start[0]) / nm_per_px + camera_dims[0]),  # X coordinate
        int((center_coords_raw[1] - start[1]) / nm_per_px + camera_dims[1])   # Y coordinate
    )

    # Add the image to the canvas within a lock to ensure thread-safe operation
    with lock:
        add_image_to_canvas(canvas, image_np, center_coords)
        

def stitch_and_display_images(frame_queue, start, end):
    """
    Continuously stitches and displays images from a queue, blending them into a larger canvas
    that includes an alpha channel for transparency. Saves the final stitched image when all frames are processed.
    
    Args:
        frame_queue (queue.Queue): A queue containing tuples of images and their corresponding center coordinates.
        start (tuple): The (x, y) starting coordinates of the scan area in nanometers.
        end (tuple): The (x, y) ending coordinates of the scan area in nanometers.
    """

    # Assuming square frames
    frame_dims = min(camera_dims)  
    
    # Calculate the size of the output canvas based on the scan area and the camera dimensions
    output_size = [
        int((end[0] - start[0]) / nm_per_px + frame_dims * 2.5),  # Width of the canvas
        int((end[1] - start[1]) / nm_per_px + frame_dims * 2.5)   # Height of the canvas
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

    save_image(canvas, "Images/final_downsampled.png", 10)

    print("Save complete")

    post_processing(canvas, start)

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
        self.area = self.area_px * (nm_per_px**2)
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
        


def post_processing(canvas, start, contrast=2, threshold=100):
    t1 = time.time()
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

    # Find contours on the downscaled post-processed image
    scaled_contours, _ = cv2.findContours(post_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale up the contours to match the original resolution
    contours = [contour * downscale_factor for contour in scaled_contours]

    for i, contour in enumerate(contours):
        # Save the bounding box of the monolayer
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding to the bounding box
        x_start = max(x - monolayer_crop_padding, 0)
        y_start = max(y - monolayer_crop_padding, 0)
        x_end = min(x + w + monolayer_crop_padding, canvas.shape[1])
        y_end = min(y + h + monolayer_crop_padding, canvas.shape[0]) 

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
    print("Saving image with monolayers...")
    save_image(contour_image, "Images/contour.png", 5)
    print("Saved!")

    # Sort monolayers by area removing any which are too small
    monolayers = [layer for layer in monolayers if layer.area_um >= downscale_factor]
    monolayers.sort(key=operator.attrgetter('area'))

    for i, layer in enumerate(monolayers):
        print(f"{i+1}: Area: {layer.area_um:.0f} um^2,  Centre: {layer.position}      Entropy: {layer.smoothed_entropy:.2f}, TV Norm: {layer.total_variation_norm:.2f}, Local Intensity Variance: {layer.local_intensity_variance:.2f}, CNR: {layer.contrast_to_noise_ratio:.2f}, Skewness: {layer.skewness:.2f}")
        cv2.imwrite(f"Monolayers/{i+1}.png", layer.image)

    while True:
        try:
            n = int(input("Go To Monolayer: "))-1
        except ValueError:
            print("Please enter a valid monolayer number")
        if n in range(0, len(monolayers)):
            move(mcm301obj, image_to_stage(monolayers[n].position, start))
        else:
            print("Please enter a valid monolayer number")


def alg(mcm301obj, image_queue, frame_queue, start, end):
    """
    Runs the main scanning algorithm to capture and process images across a defined area.
    
    Args:
        mcm301obj (MCM301): The stage controller object.
        image_queue (queue.Queue): Queue containing images captured by the camera.
        frame_queue (queue.Queue): Queue to store images along with their positions for stitching.
        start (tuple): Starting coordinates (x, y) in nanometers.
        end (tuple): Ending coordinates (x, y) in nanometers.
    """
    
    def capture_and_store_frame(x, y):
        """
        Captures a frame from the image queue and stores it with its position in the frame queue.
        
        Args:
            x (int): The x-coordinate in nanometers.
            y (int): The y-coordinate in nanometers.
        """
        time.sleep(camera_properties["exposure"]/1e6)


        ################################################################################################################################### Change this back
        frame = image_queue.get(timeout=1000)
        r = random.randint(-12, 0)
        if r > 0:
            frame = Image.open(f"Images/test_image{r}.jpg")
        frame_queue.put((frame, (x, y)))

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
            x += dist * direction
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
        y += dist
        move(mcm301obj, (x, y))
        
        # Reverse direction for the next line scan
        direction *= -1
    scan_line(x, y, direction)

    print("\nImage capture complete!")
    print("Waiting for image processing to complete...")
    frame_queue.put(None)


   
""" Main
When run as a script, a simple Tkinter app is created with just a LiveViewCanvas widget.
"""
if __name__ == "__main__":
    print(f'Using device: {accel_device} for image processing.')
    
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        with sdk.open_camera(camera_list[0]) as camera:

            # create generic Tk App with just a LiveViewCanvas widget
            print("Generating app...")
            root = tk.Tk()
            root.title(camera.name)
            image_acquisition_thread = ImageAcquisitionThread(camera, rotation_angle=270)
            camera_widget = LiveViewCanvas(parent=root, image_queue=image_acquisition_thread.get_output_queue())      
       
            print("Setting camera parameters...")

            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.arm(2)

            camera.gain = camera_properties["gain"]
            camera.exposure_time_us = camera_properties["exposure"]

            camera_dims = [camera.image_width_pixels, camera.image_height_pixels]
            dist = int(min(camera_dims) * nm_per_px * (1-image_overlap))

            camera.issue_software_trigger()

            print("Starting image acquisition thread...")
            image_acquisition_thread.start()
            image_queue = image_acquisition_thread.get_output_queue()

            frame_queue = queue.Queue()

            mcm301obj = stage_setup()
            start, end = get_scan_area(mcm301obj)

            stitching_thread = threading.Thread(target=stitch_and_display_images, args=(frame_queue, start, end))
            stitching_thread.start()

            alg_thread = threading.Thread(target=alg, args=(mcm301obj, image_queue, frame_queue, start, end))
            alg_thread.start()

            root.mainloop()

            print("Waiting for image acquisition thread to finish...")
            image_acquisition_thread.stop()
            image_acquisition_thread.join()
            # stitching_thread.stop() # This needs a stop function
            stitching_thread.join()

            # cv2.destroyAllWindows()
            print("Closing resources...")

    print("App terminated. Goodbye!")