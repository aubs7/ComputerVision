import cv2 as cv
import numpy as np

img = cv.imread('Photos/sample.png')

#Resize the image
width = 750
height = 950 
img2 = cv.resize(img, (width, height))

# Apply Averaging filter 
blur_avg = cv.blur(img2, (5, 5)) 

# Apply Gaussian filter 
blur_gaussian = cv.GaussianBlur(img2, (5, 5), 0) 

# Sharpening kernel 
kernel_sharpening = np.array([[-1, -1, -1],  
                              [-1,  9, -1],  
                              [-1, -1, -1]])

# Apply sharpening 
sharpened = cv.filter2D(img2, -1, kernel_sharpening) 

gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 

# Sobel edge detection 
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5) 
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5) 

# Convert to uint8 
sobel_x = cv.convertScaleAbs(sobel_x) 
sobel_y = cv.convertScaleAbs(sobel_y) 

# Canny edge detection 
edges = cv.Canny(gray, 100, 200)

# Convert to grayscale 
gray_image = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 

# Apply histogram equalization 
equalized_image = cv.equalizeHist(gray_image)

# Display results 
cv.imshow('Original', img2) 
cv.imshow('Averaging Blur', blur_avg) 
cv.imshow('Gaussian Blur', blur_gaussian) 
cv.imshow('Sharpened Image', sharpened) 
cv.imshow('Sobel X', sobel_x) 
cv.imshow('Sobel Y', sobel_y) 
cv.imshow('Canny Edges', edges) 
cv.imshow('Original Grayscale Image', gray_image) 
cv.imshow('Histogram Equalized Image', equalized_image) 

cv.waitKey(0) 
cv.destroyAllWindows() 