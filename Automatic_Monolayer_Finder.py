try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

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

# These bits indicate that the stage is no longer moving.
confirmation_bits = (2147484928, 2147484930)

dist = 343200
camera_dims = (2448, 2048)
nm_per_px = 171.6


# TODO
# FIX Nasty alg layout

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


def get_scan_area(mcm301obj):
    input("Please move the stage to one corner of the sample. Press ENTER when complete")
    x_1, y_1 = get_pos(mcm301obj, (4, 5))
    input("Please move the stage to the opposite corner of the sample. Press ENTER when complete")
    x_2, y_2 = get_pos(mcm301obj, (4, 5))
    start = [min(x_1, x_2), min(y_1, y_2)]
    end = [max(x_1, x_2), max(y_1, y_2)]
    return start, end

def add_image_to_canvas(canvas, image, center_coords, alpha=0.5):
    """
    Adds an image to the canvas at the specified center coordinates with full opacity.
    Only overlapping regions are blended using transparency.

    Args:
        canvas (numpy.ndarray): The canvas where the image will be overlaid.
        image (numpy.ndarray): The image to be added.
        center_coords (tuple): The (x, y) coordinates of the center of the image on the canvas.
        alpha (float): The blending factor for overlapping regions, ranging from 0 (transparent) to 1 (opaque).

    Returns:
        numpy.ndarray: The updated canvas with the image blended in.
    """

    img_height, img_width = image.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]

    # Calculate start and end coordinates for the overlay
    x_center, y_center = center_coords
    x_start = max(0, x_center - img_width // 2)
    y_start = max(0, y_center - img_height // 2)
    x_end = min(canvas_width, x_start + img_width)
    y_end = min(canvas_height, y_start + img_height)

    # Calculate the region of the canvas and image that will be used
    x_offset = x_start - (x_center - img_width // 2)
    y_offset = y_start - (y_center - img_height // 2)
    region_width = x_end - x_start
    region_height = y_end - y_start

    # Extract the relevant regions from the canvas and the image
    canvas_region = canvas[y_start:y_end, x_start:x_end]
    image_region = image[y_offset:y_offset + region_height, x_offset:x_offset + region_width]

    # Create masks for blending
    mask_canvas = (canvas_region > 0).astype(np.uint8)
    mask_image = (image_region > 0).astype(np.uint8)

    # Blend only the overlapping regions
    blended_region = cv2.addWeighted(canvas_region, 1 - alpha, image_region, alpha, 0)

    # Overlay the image with full opacity
    canvas[y_start:y_end, x_start:x_end] = np.where(mask_image, image_region, canvas_region)

    # Apply the blended region only where both images overlap
    overlap_mask = mask_canvas & mask_image
    canvas[y_start:y_end, x_start:x_end] = np.where(overlap_mask, blended_region, canvas[y_start:y_end, x_start:x_end])

    return canvas




def stitch_and_display_images(frame_queue, start, end):
    """
    Continuously stitches and displays images from a queue, blending them into a larger canvas.

    Args:
        frame_queue (queue.Queue): A queue containing tuples of images and their corresponding center coordinates.
        start (tuple): The (x, y) starting coordinates of the scan area in nanometers.
        end (tuple): The (x, y) ending coordinates of the scan area in nanometers.
    
    This function extracts images from a queue, calculates their correct positions on a larger canvas based on their 
    center coordinates, and then blends them onto the canvas using the `add_image_to_canvas` function. The function 
    continues running in a loop, updating the canvas as new images are received.
    """

    # Calculate the size of the output canvas based on the scan area and the camera dimensions
    output_size = [
        int((end[0] - start[0]) / nm_per_px + camera_dims[0] * 2),  # Width of the canvas
        int((end[1] - start[1]) / nm_per_px + camera_dims[1] * 2)   # Height of the canvas
    ]
    
    # Initialize the canvas with zeros (black background) with the calculated size
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Variable to store the previous state of the canvas (if needed for future enhancements)
    previous_canvas = None

    while True:
        # Collect a batch of frames from the queue
        batch_frames = []
        while not frame_queue.empty():
            item = frame_queue.get()
            if item is None:
                break  # Exit the loop if a None is received
            batch_frames.append(item)

        # If no frames were collected, continue to the next iteration
        if not batch_frames:
            continue

        # Process each frame in the collected batch
        for image, center_coords_raw in batch_frames:
            # Convert the image to a numpy array
            image_np = np.array(image)
            
            # Calculate the coordinates on the canvas where the image center should be placed
            center_coords = (
                int((center_coords_raw[0] - start[0]) / nm_per_px + camera_dims[0]),  # X coordinate
                int((center_coords_raw[1] - start[1]) / nm_per_px + camera_dims[1])   # Y coordinate
            )
            
            # Rotate the image 90 degrees clockwise (if required for alignment)
            image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
            
            # Add the image to the canvas, blending it with any existing images
            canvas = add_image_to_canvas(canvas, image_np, center_coords)
            cv2.imwrite("stitch.jpg", canvas)





def alg(mcm301obj, image_queue, frame_queue, start, end):
    move_and_wait(mcm301obj, start)
    x, y = start
    direction = 1
    while get_pos(mcm301obj, stages=(5,))[0] < end[1]:
        while get_pos(mcm301obj, stages=(4,))[0] < end[0]:
            time.sleep(0.3)
            frame = image_queue.get(timeout=1000)
            frame_queue.put((frame, (x, y)))
            x += dist*direction
            move_and_wait(mcm301obj, (x, y))
        time.sleep(0.3)
        frame = image_queue.get(timeout=1000)
        frame_queue.put((frame, (x, y)))
        y += dist
        move_and_wait(mcm301obj, (x, y))
        direction *= -1
        while get_pos(mcm301obj, stages=(4,))[0] > start[0]:
            time.sleep(0.3)
            frame = image_queue.get(timeout=1000)
            frame_queue.put((frame, (x, y)))
            x += dist*direction
            move_and_wait(mcm301obj, (x, y))
        time.sleep(0.3)
        frame = image_queue.get(timeout=1000)
        frame_queue.put((frame, (x, y)))
        y += dist
        move_and_wait(mcm301obj, (x, y))
        direction *= -1


    
""" Main

When run as a script, a simple Tkinter app is created with just a LiveViewCanvas widget. 

"""
if __name__ == "__main__":
    with TLCameraSDK() as sdk:
        camera_list = sdk.discover_available_cameras()
        with sdk.open_camera(camera_list[0]) as camera:

            # create generic Tk App with just a LiveViewCanvas widget
            print("Generating app...")
            root = tk.Tk()
            root.title(camera.name)
            image_acquisition_thread = ImageAcquisitionThread(camera)
            camera_widget = LiveViewCanvas(parent=root, image_queue=image_acquisition_thread.get_output_queue())       
        
            print("Setting camera parameters...")
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.arm(2)
            camera.issue_software_trigger()

            print("Starting image acquisition thread...")
            image_acquisition_thread.start()
            image_queue = image_acquisition_thread.get_output_queue()
            frame = image_queue.get(timeout=1000)
            frame.save("test.jpg")

            frame_queue = queue.Queue()

            mcm301obj = stage_setup()
            start, end = get_scan_area(mcm301obj)

            stitching_thread = threading.Thread(target=stitch_and_display_images, args=(frame_queue, start, end))
            stitching_thread.start()

            alg_thread = threading.Thread(target=alg, args=(mcm301obj, image_queue, frame_queue, start, end))
            alg_thread.start()

            print("App starting")
            root.mainloop()

            print("Waiting for image acquisition thread to finish...")
            image_acquisition_thread.stop()
            image_acquisition_thread.join()

            stitching_thread.join()

            cv2.destroyAllWindows()

            print("Closing resources...")

    print("App terminated. Goodbye!")
