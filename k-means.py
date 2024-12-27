import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

img = cv.imread('Photos/sample3.jpg')

width = 600
height = 950 

img2 = cv.resize(img, (width, height))
image = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
 
plt.imshow(image)


# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
k = 6
retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
 
# Convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# Reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))


# Display result
plt.imshow(segmented_image)
cv.imshow("Original Image", img2)
cv.imshow("K-Means Clustering", segmented_image)
cv.waitKey(0) 
cv.destroyAllWindows()