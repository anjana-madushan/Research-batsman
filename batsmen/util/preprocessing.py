import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Apply Gaussian Blur to remove blur
    # blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    
    # Apply Median Blur to remove salt-and-pepper noise
    # median_blurred_image = cv2.medianBlur(image, 5)
    
    # # Perform resizing and cropping if necessary
    # # resized_image = cv2.resize(median_blurred_image, (224, 224))
    
    # # # Normalize pixel values to range [0, 1]
    # # normalized_image = median_blurred_image / 255.0
    
    # # Convert image to 8-bit unsigned integer depth
    # uint8_image = (median_blurred_image * 255).astype(np.uint8)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')  # Hide axes
    # plt.show()
    
    return image