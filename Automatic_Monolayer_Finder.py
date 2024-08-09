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

def compute_canvas_dimensions(start, end):
    width = int((end[0] - start[0]) * 171.6 * 1.05)
    height = int((end[1] - start[1]) * 171.6 * 1.05)
    return width, height

def add_image_to_canvas(canvas, image, center_coords):
    img_height, img_width = image.shape[:2]
    x_center, y_center = center_coords

    x_start = max(0, x_center - img_width // 2)
    y_start = max(0, y_center - img_height // 2)
    
    x_end = min(canvas.shape[1], x_start + img_width)
    y_end = min(canvas.shape[0], y_start + img_height)
    
    canvas[y_start:y_end, x_start:x_end] = np.where(image[:(y_end - y_start), :(x_end - x_start)] > 0,
                                                     image[:(y_end - y_start), :(x_end - x_start)],
                                                     canvas[y_start:y_end, x_start:x_end])
    return canvas

def stitch_and_display_images(frame_queue, canvas, fixed_size=(500, 500), update_interval=5):
    update_counter = 0

    while True:
        while not frame_queue.empty():
            image, center_coords = frame_queue.get()
            if image is None:
                return

            image_np = cv2.rotate(np.array(image), cv2.ROTATE_90_CLOCKWISE)
            canvas = add_image_to_canvas(canvas, image_np, center_coords)

            update_counter += 1
            if update_counter % update_interval == 0:
                canvas_height, canvas_width = canvas.shape[:2]
                scale_factor = min(fixed_size[0] / canvas_width, fixed_size[1] / canvas_height, 1)
                resized_canvas = cv2.resize(canvas, (int(canvas_width * scale_factor), int(canvas_height * scale_factor)), interpolation=cv2.INTER_AREA)

                display_canvas = np.zeros((fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
                start_x = (fixed_size[0] - resized_canvas.shape[1]) // 2
                start_y = (fixed_size[1] - resized_canvas.shape[0]) // 2
                display_canvas[start_y:start_y + resized_canvas.shape[0], start_x:start_x + resized_canvas.shape[1]] = resized_canvas

                cv2.imshow('Stitched Image', display_canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

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


            mcm301obj = stage_setup()
            start, end = get_scan_area(mcm301obj)

            canvas_width, canvas_height = compute_canvas_dimensions(start, end)
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            stitching_thread = threading.Thread(target=stitch_and_display_images, args=(frame_queue, canvas, (500,500), 5), daemon=True).start()

            stitching_thread = threading.Thread(target=stitch_and_display_images, args=(frame_queue,))

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
