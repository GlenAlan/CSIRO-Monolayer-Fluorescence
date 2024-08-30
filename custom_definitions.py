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

import tkinter as tk
from PIL import Image, ImageTk
import typing
import threading
import queue
import os
import cv2


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
    cv2.imwrite(filename, scale_down_canvas(image, scale_down_factor))
    print(format_size(os.path.getsize(filename)))


class LiveViewCanvas(tk.Canvas):

    def __init__(self, parent, image_queue):
        # type: (typing.Any, queue.Queue) -> LiveViewCanvas
        self.image_queue = image_queue
        self._image_width = 0
        self._image_height = 0

        tk.Canvas.__init__(self, parent)
        self.grid(column=0, row=0, rowspan=4, columnspan=2, sticky=tk.E)
        self._get_image()

    def _get_image(self):
        try:
            image = self.image_queue.get_nowait()
            aspect = image.size[0]/image.size[1]

            # resize image
            image = image.resize((int(500*aspect),500))
            
            
            self._image = ImageTk.PhotoImage(master=self, image=image)
            if (self._image.width() != self._image_width) or (self._image.height() != self._image_height):
                # resize the canvas to match the new image size
                self._image_width = image.size[0] #self._image.width()
                self._image_height = image.size[1] #self._image.height()
                self.config(width=self._image_width, height=self._image_height)
            self.create_image(0, 0, image=self._image, anchor='nw')
        except queue.Empty:
            pass
        self.after(10, self._get_image)

class ImageAcquisitionThread(threading.Thread):
    def __init__(self, camera, rotation_angle=0):
        # type: (TLCamera, float) -> ImageAcquisitionThread
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._previous_timestamp = 0
        self._rotation_angle = rotation_angle  # New parameter for rotation

        # setup color processing if necessary
        if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            # Sensor type is not compatible with the color processing library
            self._is_color = False
        else:
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

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0  # Do not want to block for long periods of time
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_color_image(self, frame):
        # type: (Frame) -> Image
        # verify the image size
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]
        if (width != self._image_width) or (height != self._image_height):
            self._image_width = width
            self._image_height = height
            print("Image dimension change detected, image acquisition thread was updated")

        # color the image. transform_to_24 will scale to 8 bits per channel
        color_image_data = self._mono_to_color_processor.transform_to_24(
            frame.image_buffer,
            self._image_width,
            self._image_height
        )
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)
        
        # Create a PIL Image object
        pil_image = Image.fromarray(color_image_data, mode='RGB')


        # Rotate the image by the specified angle
        if self._rotation_angle != 0:
            pil_image = pil_image.rotate(self._rotation_angle, expand=True)


        return pil_image


    def _get_image(self, frame):
        # type: (Frame) -> Image
        # no coloring, just scale down image to 8 bpp and place into PIL Image object
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        pil_image = Image.fromarray(scaled_image)


        # Rotate the image by the specified angle
        if self._rotation_angle != 0:
            pil_image = pil_image.rotate(self._rotation_angle, expand=True)


        return pil_image


    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    if self._is_color:
                        pil_image = self._get_color_image(frame)
                    else:
                        pil_image = self._get_image(frame)
                    self._image_queue.put_nowait(pil_image)
            except queue.Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        print("Image acquisition has stopped")
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()