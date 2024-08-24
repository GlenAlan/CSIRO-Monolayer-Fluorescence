try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

from MCM301_COMMAND_LIB import *
import time
import random

# from custom_definitions import *

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

# These bits indicate that the stage is no longer moving.
confirmation_bits = (2147484928, 2147484930)
monolayer_crop_padding = 10

camera_dims = (2448, 2048)
nm_per_px = 171.6
image_overlap = 0.05
dist = int(min(camera_dims) * nm_per_px * (1-image_overlap))

# TODO
# FIX Nasty alg layout

def stage_setup():
    mcm301obj = MCM301()
    return mcm301obj


def move_and_wait(mcm301obj, pos, stage=(4, 5)):
    x_nm, y_nm, = pos
    print(f"Moving to {x_nm}, {y_nm}")


def get_pos(mcm301obj, stages=[4, 5, 6]):
    pos = []
    return pos


def get_scan_area(mcm301obj):
    start = 2e6, 2e6
    end = 4e6, 4e6
    return start, end


def add_image_to_canvas(canvas, image, center_coords, alpha=0.5):
    """
    Adds an image with an alpha channel to the canvas at the specified center coordinates.
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

    # Extract the relevant regions from the canvas and the cropped image
    canvas_region = canvas[y_start:y_end, x_start:x_end]
    image_region = cropped_image[y_offset:y_offset + region_height, x_offset:x_offset + region_width]

    # Ensure that the image region has an alpha channel
    if image_region.shape[2] == 3:
        image_region = cv2.cvtColor(image_region, cv2.COLOR_RGB2RGBA)

    # Perform alpha blending in the overlapping regions
    image_alpha = image_region[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
    canvas_alpha = canvas_region[:, :, 3] / 255.0

    # Calculate the combined alpha where both images overlap
    combined_alpha = image_alpha * alpha + canvas_alpha * (1 - image_alpha * alpha)
    combined_alpha[combined_alpha == 0] = 1

    # Blend only the RGB channels where the overlap occurs
    for c in range(3):  # Loop over RGB channels
        canvas_region[:, :, c] = np.where(
            canvas_alpha > 0,  # Only blend in the overlapping regions
            (image_region[:, :, c] * image_alpha * alpha + canvas_region[:, :, c] * canvas_alpha * (1 - image_alpha * alpha)) / combined_alpha,
            image_region[:, :, c]  # Directly overlay the image where no overlap occurs
        )

    # Update the alpha channel in the canvas, ensuring full opacity where the image is placed
    canvas_region[:, :, 3] = np.where(canvas_alpha > 0, combined_alpha * 255, 255)

    # Place the blended region back into the canvas
    canvas[y_start:y_end, x_start:x_end] = canvas_region

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
    global nm_per_px, camera_dims  # Use the global variables

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

    # Rotate the image 90 degrees clockwise
    image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)

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
    global nm_per_px, camera_dims  # Use the global variables
    
    # Calculate the size of the output canvas based on the scan area and the camera dimensions
    output_size = [
        int((end[0] - start[0]) / nm_per_px + camera_dims[0] * 2.5),  # Width of the canvas
        int((end[1] - start[1]) / nm_per_px + camera_dims[1] * 2.5)   # Height of the canvas
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
    cv2.imwrite("Images/final.png", canvas)
    print("Save complete")

    post_processing(canvas)

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
            self.cx, self.cy = 0, 0
        self.area_px = cv2.contourArea(contour)
        self.position = (self.cx, self.cy)
        self.area = self.area_px * (nm_per_px**2)
        self.area_um = self.area / 1e6

        self.contour = contour - np.array([[x_start, y_start]])

        self.quality = 1000 * 1/(self.compute_smoothed_entropy()+1e-12) * 1/(self.compute_tv_norm()+1e-12)

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
        
        


def post_processing(canvas, contrast=2, threshold=100):
    print("Post processing...")
    monolayers = []

    post_image = canvas.copy()
    # Convert to More red less green
    # We are in BGR but laying out in RGB
    post_image = cv2.blur(post_image, (20, 20))
    post_image = 1 * post_image[:, :, 2] + - 0.75 * post_image[:, :, 1] + -0.25 * post_image[:, :, 0]
    post_image = np.clip(post_image, 0, 255)
    post_image = post_image.astype(np.uint8)
    post_image = cv2.blur(post_image, (15, 15))
    
    # Increase contrast
    post_image = cv2.convertScaleAbs(post_image, alpha=contrast, beta=0)

    # Remove pixels below a certain brightness threshold
    #_, post_image = cv2.threshold(post_image, threshold, 255, cv2.THRESH_TOZERO)
    _, post_image = cv2.threshold(post_image, threshold, 255, cv2.THRESH_BINARY)

    print("Saving post processed image...")
    cv2.imwrite("Images/processed.png", post_image)
    print("Saved!")

    # Draw contours on the canvas
    contour_image = canvas.copy()

    print("Locating Monolayers...")
    # Find contours
    contours, _ = cv2.findContours(post_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        x_start = max(x - monolayer_crop_padding, 0)
        y_start = max(y - monolayer_crop_padding, 0)
        x_end = min(x + w + monolayer_crop_padding, canvas.shape[1])
        y_end = min(y + h + monolayer_crop_padding, canvas.shape[0]) 
        image_section = canvas[y_start:y_end, x_start:x_end]

        monolayers.append(Monolayer(contour, image_section, (x_start, y_start)))

        cx, cy = monolayers[-1].position
        
        #print(f'Monolayer {i+1}: Center ({cx}, {cy}), Area: {area}')
        contour_image = cv2.circle(contour_image, (cx, cy), 5, color=(0, 0, 0, 255), thickness=-1)
    
    cv2.drawContours(contour_image, contours, -1, (255, 255, 0, 255), 4)
    
    # Display the final image with contours
    print("Saving image with monolayers...")
    cv2.imwrite("Images/contour.png", contour_image)
    print("Saved!")

    monolayers.sort(key=operator.attrgetter('area'))
    for i, layer in enumerate(monolayers):
        print(f"{i+1}: Area: {layer.area_um:.0f} um^2,  Centre: {layer.position}, Quality: {layer.quality:.2f}")
        cv2.imwrite(f"Monolayers/{i+1}.png", layer.image)


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
        time.sleep(0.05)


        ################################################################################################################################### Change this back
        r = random.randint(1, 4)
        frame = Image.open(f"Images/test_image{r}.jpg")
        frame_queue.put((frame, (x, y)))

        ###################################################################################################################################

    def scan_line(x, y, direction, x_end):
        """
        Scans a single line in the current direction, capturing images along the way.
        
        Args:
            x (int): The current x-coordinate in nanometers.
            y (int): The current y-coordinate in nanometers.
            direction (int): The scanning direction (1 for forward, -1 for backward).
            x_end (int): The x-coordinate to stop scanning at.
            
        Returns:
            int: The updated x-coordinate after completing the line scan.
        """
        while (direction == 1 and x < x_end) or (direction == -1 and x > start[0]):
            capture_and_store_frame(x, y)
            x += dist * direction
            move_and_wait(mcm301obj, (x, y))
        
        # Capture final frame at the line end
        capture_and_store_frame(x, y)
        
        return x
    
    # Start scanning
    move_and_wait(mcm301obj, start)
    x, y = start
    direction = 1
    
    while y < end[1]:
        x = scan_line(x, y, direction, end[0])
        
        # Move to the next line
        y += dist
        move_and_wait(mcm301obj, (x, y))
        
        # Reverse direction for the next line scan
        direction *= -1

    print("\nImage capture complete!")
    print("Waiting for image processing to complete...")
    frame_queue.put(None)

   
""" Main
When run as a script, a simple Tkinter app is created with just a LiveViewCanvas widget.
"""
if __name__ == "__main__":
    # create generic Tk App with just a LiveViewCanvas widget
    print("Generating app...")
    root = tk.Tk()


    print("Starting image acquisition thread...")

    frame_queue = queue.Queue()

    mcm301obj = stage_setup()
    start, end = get_scan_area(mcm301obj)

    stitching_thread = threading.Thread(target=stitch_and_display_images, args=(frame_queue, start, end))
    stitching_thread.start()

    alg_thread = threading.Thread(target=alg, args=(mcm301obj, None, frame_queue, start, end))
    alg_thread.start()

    root.mainloop()

    print("Waiting for image acquisition thread to finish...")

    # stitching_thread.stop() # This needs a stop function
    stitching_thread.join()

    cv2.destroyAllWindows()
    print("Closing resources...")

    print("App terminated. Goodbye!")