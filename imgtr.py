import cv2 as cv
import numpy as np 

img = cv.imread('Photos/sample2.jpg')

height = 600
width = 950 
img2 = cv.resize(img, (width, height))

cv.circle(img2, (50, 50), 5, (0, 0, 255), -1) 



# Translation Matrix
tx, ty = 70, 30 
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]]) 

# Apply translation 
translated_img = cv.warpAffine(img2, translation_matrix, (img2.shape[1], 
img2.shape[0])) 

cv.circle(translated_img, (50 + tx, 50 + ty), 5, (255, 0, 0), -1) 



# Rotation Matrix
center = (img2.shape[1] // 2, img2.shape[0] // 2) 
angle = 45 
rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0) 

# Apply rotation 
rotated_img = cv.warpAffine(img2, rotation_matrix, (img2.shape[1], img2.shape[0])) 


cv.circle(rotated_img, (center[0] - 3, center[1] + 2), 5, (0, 255, 0), -1) 



# Scaling
sx, sy = 3, 3
scaled_img = cv.resize(img2, None, fx=sx, fy=sy) 

cv.circle(scaled_img, (50 * sx, 50 * sy), 10, (0, 255, 255), -1)



# Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]]) 
pts2 = np.float32([[10, 100], [200, 50], [100, 250]]) 

# Calculate the matrix
affine_matrix = cv.getAffineTransform(pts1, pts2) 

# Apply the affine transformation 
affine_transformed_img = cv.warpAffine(img2, affine_matrix, (img2.shape[1], img2.shape[0])) 



# Homography Transformation 
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]]) 
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [220, 220]]) 

# Calculate the Homography matrix 
homography_matrix, _ = cv.findHomography(pts1, pts2) 

# Apply Homography 
projective_transformed_img = cv.warpPerspective(img2, homography_matrix, (img2.shape[1], img2.shape[0])) 



# Display results 
cv.imshow('Original Image', img2) 
cv.imshow('Translated Image', translated_img) 
cv.imshow('Rotated Image', rotated_img) 
cv.imshow('Scaled Image', scaled_img) 
cv.imshow('Affine Transformed Image', affine_transformed_img)
cv.imshow('Projective Transformed Image', projective_transformed_img) 
cv.waitKey(0) 
cv.destroyAllWindows()