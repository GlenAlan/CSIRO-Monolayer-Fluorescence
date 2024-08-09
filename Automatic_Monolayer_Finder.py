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

confirmation_bits = (2147484928, 2147484930) # These bits indicate that the stage is no longer moving.

dist = 343200


# TODO
# FIX Nasty alg layout

def stage_setup():
    mcm301obj = MCM301()

    devs = MCM301.list_devices()
    print(devs)
    if len(devs) <= 0:
        print('There is no devices connected')
        exit()
    device_info = devs[0]
    sn = device_info[0]
    print("connect ", sn)
    hdl = mcm301obj.open(sn, 115200, 3)
    if hdl < 0:
        print("open ", sn, " failed. hdl is ", hdl)
        exit()
    if mcm301obj.is_open(sn) == 0:
        print("MCM301IsOpen failed")
        mcm301obj.close()
        exit()

    for stage_num in (4,5):
        print(f"Homing stage {stage_num}")
        mcm301obj.home(stage_num)

    bits_x, bits_y = [0], [0]
    while bits_x[0] not in confirmation_bits or bits_y[0] not in confirmation_bits:
        mcm301obj.get_mot_status(4, [0], bits_x)
        mcm301obj.get_mot_status(5, [0], bits_y)
        # print(f"x: {bits_x}, y:{bits_y}")
    
    print("Homing complete")
    print("Stage setup complete")

    return mcm301obj

def move_and_wait(mcm301obj, pos, stage=(4, 5)):
    x_nm, y_nm, = pos
    print(f"Moving to {x_nm}, {y_nm}")
    x, y = [0], [0]
    stage_x, stage_y = stage
    encoder_val_x, bits_x = [0], [0]
    encoder_val_y, bits_y = [0], [0]

    mcm301obj.convert_nm_to_encoder(stage_x, x_nm, x)
    mcm301obj.convert_nm_to_encoder(stage_y, y_nm, y)

    mcm301obj.move_absolute(stage_x, x[0])
    mcm301obj.move_absolute(stage_y, y[0])


    while bits_x[0] not in confirmation_bits or bits_y[0] not in confirmation_bits:
        mcm301obj.get_mot_status(stage_x, encoder_val_x, bits_x)
        mcm301obj.get_mot_status(stage_y, encoder_val_y, bits_y)
        # print(f"x: {bits_x}, y:{bits_y}")

def get_pos(mcm301obj, stages=[4, 5, 6]):
    pos = []
    for stage in stages:
        encoder_val, nm = [0], [0]
        mcm301obj.get_mot_status(stage, encoder_val, [0])
        mcm301obj.convert_encoder_to_nm(stage, encoder_val[0], nm)
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

def add_image_to_canvas(canvas, image, center_coords):
    img_height, img_width = image.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]

    x_center, y_center = center_coords
    x = max(0, x_center - img_width // 2)
    y = max(0, y_center - img_height // 2)

    new_canvas_height = max(canvas_height, y + img_height)
    new_canvas_width = max(canvas_width, x + img_width)

    if new_canvas_height > canvas_height or new_canvas_width > canvas_width:
        new_canvas = np.zeros((new_canvas_height, new_canvas_width, 3), dtype=np.uint8)
        new_canvas[:canvas_height, :canvas_width] = canvas
        canvas = new_canvas

    img_cropped = image[:canvas.shape[0] - y, :canvas.shape[1] - x]
    roi = canvas[y:y + img_cropped.shape[0], x:x + img_cropped.shape[1]]

    # Blending using addWeighted
    alpha_image = np.any(img_cropped != 0, axis=-1).astype(np.float32)
    alpha_canvas = np.any(roi != 0, axis=-1).astype(np.float32)

    overlap = alpha_image * alpha_canvas
    alpha_image[overlap > 0] /= 2
    alpha_canvas[overlap > 0] /= 2

    blended = cv2.addWeighted(roi, 0.5, img_cropped, 0.5, 0)
    non_overlap_mask = alpha_image > alpha_canvas
    blended[non_overlap_mask] = img_cropped[non_overlap_mask]

    canvas[y:y + blended.shape[0], x:x + blended.shape[1]] = blended

    return canvas


def stitch_and_display_images(frame_queue):
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    fixed_size = (500, 500)
    previous_canvas = None

    while True:
        item = frame_queue.get()
        if item is None:
            break

        image, center_coords = item
        image_np = np.array(image)
        image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        canvas = add_image_to_canvas(canvas, image_np, center_coords)

        # Only update the display if the canvas has changed
        if previous_canvas is None or not np.array_equal(previous_canvas, canvas):
            canvas_height, canvas_width = canvas.shape[:2]
            scale_factor = min(fixed_size[0] / canvas_width, fixed_size[1] / canvas_height, 1)
            new_width = int(canvas_width * scale_factor)
            new_height = int(canvas_height * scale_factor)
            resized_canvas = cv2.resize(canvas, (new_width, new_height))
            
            display_canvas = np.zeros((fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
            start_x = (fixed_size[0] - new_width) // 2
            start_y = (fixed_size[1] - new_height) // 2
            display_canvas[start_y:start_y + new_height, start_x:start_x + new_width] = resized_canvas

            cv2.imshow('Stitched Image', display_canvas)
            cv2.waitKey(30)  # Update display every 30 milliseconds

            previous_canvas = canvas.copy()

    cv2.destroyAllWindows()




def alg(mcm301obj, image_queue, frame_queue, start, end):
    move_and_wait(mcm301obj, start)
    x, y = start
    direction = 1
    while get_pos(mcm301obj, stages=(5,))[0] < end[1]:
        while get_pos(mcm301obj, stages=(4,))[0] < end[0]:
            time.sleep(0.3)
            frame = image_queue.get(timeout=1000)
            frame_queue.put((frame, (int(x/171.6), int(y/171.6))))
            x += dist*direction
            move_and_wait(mcm301obj, (x, y))
        time.sleep(0.3)
        frame = image_queue.get(timeout=1000)
        frame_queue.put((frame, (int(x/171.6), int(y/171.6))))
        y += dist
        move_and_wait(mcm301obj, (x, y))
        direction *= -1
        while get_pos(mcm301obj, stages=(4,))[0] > start[0]:
            time.sleep(0.3)
            frame = image_queue.get(timeout=1000)
            frame_queue.put((frame, (int(x/171.6), int(y/171.6))))
            x += dist*direction
            move_and_wait(mcm301obj, (x, y))
        time.sleep(0.3)
        frame = image_queue.get(timeout=1000)
        frame_queue.put((frame, (int(x/171.6), int(y/171.6))))
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

            frame_queue = queue.Queue()

            stitching_thread = threading.Thread(target=stitch_and_display_images, args=(frame_queue,))

            stitching_thread.start()


            mcm301obj = stage_setup()
            start, end = get_scan_area(mcm301obj)

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
