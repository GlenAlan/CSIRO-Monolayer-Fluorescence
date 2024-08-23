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
import operator

# These bits indicate that the stage is no longer moving.
confirmation_bits = (2147484928, 2147484930)
monolayer_crop_padding = 10

dist = 343000
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
    print("Stage setup complete\n")

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

    img_height, img_width = image.shape[:2]
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

    # Extract the relevant regions from the canvas and the image
    canvas_region = canvas[y_start:y_end, x_start:x_end]
    image_region = image[y_offset:y_offset + region_height, x_offset:x_offset + region_width]

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


def stitch_and_display_images(frame_queue, start, end):
    """
    Continuously stitches and displays images from a queue, blending them into a larger canvas
    that includes an alpha channel for transparency. Saves the final stitched image when all frames are processed.
    
    Args:
        frame_queue (queue.Queue): A queue containing tuples of images and their corresponding center coordinates.
        start (tuple): The (x, y) starting coordinates of the scan area in nanometers.
        end (tuple): The (x, y) ending coordinates of the scan area in nanometers.
        output_filename (str): The filename where the final stitched image will be saved.
    """
    
    # Calculate the size of the output canvas based on the scan area and the camera dimensions
    output_size = [
        int((end[0] - start[0]) / nm_per_px + camera_dims[0] * 2.5),  # Width of the canvas
        int((end[1] - start[1]) / nm_per_px + camera_dims[1] * 2.5)   # Height of the canvas
    ]
    
    # Initialize the canvas
    canvas = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)  # 4 channels (RGBA)

    while True:
        # Get the next item from the frame queue
        item = frame_queue.get()
        
        if item is None:
            # Signal received that all frames are processed
            print("Image stitching complete")
            print("Saving final image...")
            cv2.imwrite("Images/final.png", canvas)
            print("Save complete")
            break
        
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
        
        # Add the image to the canvas
        canvas = add_image_to_canvas(canvas, image_np, center_coords)
        cv2.imwrite("Images/stitch.png", canvas)

    post_processing(canvas)

class Monolayer:
    def __init__(self, contour, image):
        self.image = image

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

        monolayers.append(Monolayer(contour, image_section))

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
        print(f"{i+1}: Area: {layer.area:.0f} um,  Centre: {layer.position}")
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
        time.sleep(0.5)


        ################################################################################################################################### Change this back
        frame = image_queue.get(timeout=1000)
        r = random.randint(0, 4)
        if r != 0:
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

            cv2.destroyAllWindows()
            print("Closing resources...")

    print("App terminated. Goodbye!")