import cv2
import numpy as np
from PIL import Image
import threading
import queue
import time

def create_blending_mask(img_shape, blend_width):
    mask = np.zeros(img_shape[:2], dtype=np.float32)
    h, w = img_shape[:2]
    
    if blend_width > 0:
        for i in range(w):
            mask[:, i] = min(i / blend_width, 1.0)
        
        for j in range(h):
            mask[j, :] = np.minimum(mask[j, :], min(j / blend_width, 1.0))
    
    if len(img_shape) == 3 and img_shape[2] == 3:
        mask = np.dstack([mask] * 3)
    
    return mask

def add_image_to_canvas(canvas, image, center_coords, max_blend_width=50):
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

    roi_mask = np.any(roi != 0, axis=-1)
    img_mask = np.any(img_cropped != 0, axis=-1)
    
    combined_mask = roi_mask & img_mask

    overlap_height = np.sum(combined_mask, axis=0).max() if combined_mask.any() else 0
    overlap_width = np.sum(combined_mask, axis=1).max() if combined_mask.any() else 0
    blend_width = min(overlap_width, overlap_height, max_blend_width) if combined_mask.any() else 0

    blend_mask = create_blending_mask(img_cropped.shape, blend_width)

    if combined_mask.any():
        combined = (roi * (1 - blend_mask) + img_cropped * blend_mask).astype(np.uint8)
    else:
        combined = img_cropped

    non_overlap_mask = img_mask & ~roi_mask
    combined[non_overlap_mask] = img_cropped[non_overlap_mask]

    canvas[y:y + combined.shape[0], x:x + combined.shape[1]] = combined

    return canvas

def stitch_images(image_queue, canvas_queue):
    canvas = np.zeros((500, 500, 3), dtype=np.uint8)
    
    while True:
        item = image_queue.get()
        if item is None:
            break

        image, center_coords = item
        image_np = np.array(image)
        canvas = add_image_to_canvas(canvas, image_np, center_coords)
        canvas_queue.put(canvas.copy())
    
    canvas_queue.put(None)

def producer(image_queue, image_paths, center_coordinates):
    for path, center in zip(image_paths, center_coordinates):
        image = Image.open(path).convert("RGB")
        image_queue.put((image, center))
        time.sleep(5)
    image_queue.put(None)

def display_canvas(canvas_queue):
    fixed_size = (500, 500)
    while True:
        canvas = canvas_queue.get()
        if canvas is None:
            break
        
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
        cv2.waitKey(1)

if __name__ == "__main__":
    image_paths = ['Images/Screenshot 2024-07-16 154534.png']*5
    center_coordinates = [(0, 0), (int(772*1.5)-10, 0), (0, int(630*1.5)-10), (int(772*1.5)-10, int(630*1.5)-10), (int(772*2.5)-10, int(630*1.5)-10)]

    image_queue = queue.Queue()
    canvas_queue = queue.Queue()

    producer_thread = threading.Thread(target=producer, args=(image_queue, image_paths, center_coordinates))
    stitching_thread = threading.Thread(target=stitch_images, args=(image_queue, canvas_queue))
    display_thread = threading.Thread(target=display_canvas, args=(canvas_queue,))

    producer_thread.start()
    stitching_thread.start()
    display_thread.start()

    producer_thread.join()
    stitching_thread.join()
    display_thread.join()

    cv2.destroyAllWindows()
