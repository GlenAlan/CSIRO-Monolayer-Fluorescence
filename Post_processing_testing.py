import cv2
import numpy as np


def resize_image(image, target_width=1200):
    """
    Resizes the image to the target width while maintaining the aspect ratio.
    
    Args:
        image (numpy.ndarray): The input image.
        target_width (int): The desired width of the resized image.
        
    Returns:
        numpy.ndarray: The resized image.
    """
    height, width = image.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image


def post_processing(canvas, contrast=2.5, threshold=100):
    post_image = canvas.copy()
    # Convert to More red less green
    # We are in 
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

    # Draw contours on the canvas
    contour_image = canvas.copy()

    print("Locating Monolayers... \n")
    # Find contours
    contours, _ = cv2.findContours(post_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0
        area = cv2.contourArea(contour)
        print(f'Monolayer {i+1}: Center ({cx}, {cy}), Area: {area}')
        contour_image = cv2.circle(contour_image, (cx, cy), 7, color=(255, 0, 255, 255), thickness=-1)
    
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0, 255), 3)
    
    return post_image, contour_image


if __name__ == "__main__":
    # Load the image you want to post-process
    input_image_path = "Images/final.png"  # Replace with your image path
    canvas = cv2.imread(input_image_path)


    image_np = np.array(canvas)
    if image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)



    # Apply post-processing
    processed_image, contour_image = post_processing(image_np)

    processed_image = resize_image(processed_image)
    contour_image = resize_image(contour_image)



    # Optionally, display the image
    cv2.imshow("Processed Image", processed_image)
    cv2.imshow("Contour Image", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





