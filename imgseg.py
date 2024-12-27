import cv2 as cv
import numpy as np 

img = cv.imread('Photos/sample3.jpg', 0)

width = 600
height = 950 
img2 = cv.resize(img, (width, height))

# Apply Otsu's thresholding 
ret, thresh = cv.threshold(img2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) 
 
# Apply mean adaptive thresholding 
thresh1 = cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2) 
 
# Apply Gaussian adaptive thresholding 
thresh2 = cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)


#Region Growing
def region_growing(img2, seed_point, threshold): 
    rows, cols = img2.shape 
    mask = np.zeros_like(img2) 
    visited = np.zeros_like(img2, dtype=bool)
    queue = [seed_point] 
 
    while queue: 
        x, y = queue.pop(0) 
        if not visited[x, y]: 
            visited[x, y] = True
            mask[x, y] = 1 
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] 
            for nx, ny in neighbors: 
                if (0 <= nx < rows and 0 <= ny < cols and 
                    not visited[nx, ny] and 
                    abs(int(img2[x, y]) - int(img2[nx, ny])) <= threshold): 
                    queue.append((nx, ny))

    return mask 

# Parameters for region growing
seed_point = (450, 500)  # Adjust seed point to a more homogeneous region
threshold = 6 # Increase threshold to allow better growth

# Apply region growing
mask = region_growing(img2, seed_point, threshold)

# Scale mask to 0-255 for visibility
mask_scaled = (mask * 215).astype(np.uint8)


# Display results
cv.imshow("Original Image", img2)
cv.imshow("Otsu Thresholding", thresh) 
cv.imshow("Mean Adaptive Thresholding", thresh1) 
cv.imshow("Gaussian Adaptive Thresholding", thresh2) 
cv.imshow("Region Growing", mask_scaled) 
cv.waitKey(0) 
cv.destroyAllWindows()