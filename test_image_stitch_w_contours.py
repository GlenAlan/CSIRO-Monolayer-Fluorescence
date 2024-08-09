import cv2
import numpy as np
from PIL import Image
import threading
import queue
import time

def preprocess_image(image, brightness_threshold=90, contrast_increase=1.0):
    # Convert to grayscale (ONLY RED)
    gray = 0.5 * image[:, :, 2] + 0.0 * image[:, :, 1] + 0.0 * image[:, :, 0]
    gray = gray.astype(np.uint8)
    
    # Increase contrast
    contrasted = cv2.convertScaleAbs(gray, alpha=contrast_increase, beta=0)
    
    # Remove pixels below a certain brightness threshold
    _, thresh = cv2.threshold(contrasted, brightness_threshold, 255, cv2.THRESH_TOZERO)
    blur = cv2.blur(thresh, (5, 5))

    
    return blur

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

    # Blending process
    alpha_image = np.any(img_cropped != 0, axis=-1).astype(np.float32)
    alpha_canvas = np.any(roi != 0, axis=-1).astype(np.float32)

    overlap = alpha_image * alpha_canvas
    alpha_image[overlap > 0] /= 2
    alpha_canvas[overlap > 0] /= 2

    blended = (roi * alpha_canvas[:, :, None] + img_cropped * alpha_image[:, :, None]).astype(np.uint8)

    non_overlap_mask = alpha_image > alpha_canvas
    blended[non_overlap_mask] = img_cropped[non_overlap_mask]

    canvas[y:y + blended.shape[0], x:x + blended.shape[1]] = blended

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
    
    # Save the final canvas
    cv2.imwrite('final_image.png', canvas)
    
    # Convert the final canvas to grayscale
    gray = preprocess_image(canvas)  # Apply preprocessing step
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Draw contours on the canvas
    canvas_with_contours = canvas.copy()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        area = cv2.contourArea(contour)
        print(f'Contour {i+1}: Center ({cx}, {cy}), Area: {area}')
        canvas_with_contours = cv2.circle(canvas_with_contours, (cx, cy), 4, color=(0, 0, 0), thickness=-1)
    
    cv2.drawContours(canvas_with_contours, contours, -1, (0, 255, 0), 2)
    
    # Display the final image with contours
    cv2.imshow('Contours', canvas_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def producer(image_queue, image_paths, center_coordinates):
    for path, center in zip(image_paths, center_coordinates):
        image = cv2.imread(path)
        image_queue.put((image, center))
        time.sleep(3)
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
    center_coordinates = [(0, 0), (int(772*1.5)-50, 0), (0, int(630*1.5)-50), (int(772*1.5)-50, int(630*1.5)-50), (int(772*2.5)-50, int(630*1.5)-50)]

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
