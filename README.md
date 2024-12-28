# CV Techniques using Diseased Citrus Leaves

This repository contains applications of various computer vision techniques for analyzing and detecting diseases in citrus leaves. It provides scripts and resources for image processing, transformation, segmentation, and clustering.

## Repository Structure

### 1. `Photos/`
This folder contains sample images of diseased citrus leaves. These images are essential for testing and validating the implemented computer vision techniques.

### 2. `imgpr.py`
A Python script implementing various **image processing techniques**, such as:
- **Averaging Filter**: Reduces noise by averaging pixel values.
- **Gaussian Filter**: Smoothens the image using a Gaussian function.
- **Sharpening**: Enhances edges and details in the image.
- **Sobel Edge Detection (X and Y)**: Detects edges along the horizontal and vertical axes.
- **Canny Edge Detection**: Identifies edges using multi-stage edge detection.
- **Histogram Equalization**: Improves image contrast by adjusting the intensity distribution.

### 3. `imgtr.py`
A Python script for **image transformation techniques**, including:
- **Translation**: Moves the image along X and Y axes.
- **Rotation**: Rotates the image to standardize orientations.
- **Scaling**: Resizes the image to normalize dimensions.
- **Affine Transformation**: Applies linear transformations such as rotation, scaling, and translation.
- **Projective Transformation**: Performs perspective adjustments to correct distortions.

### 4. `imgseg.py`
A Python script implementing **image segmentation techniques**, including:
- **Otsu Thresholding**: An automated method for binary image segmentation.
- **Mean Adaptive Thresholding**: Local thresholding using the mean of surrounding pixels.
- **Gaussian Adaptive Thresholding**: Local thresholding with Gaussian-weighted pixel sums.
- **Region Growing**: A segmentation method that groups pixels based on similarity, growing regions iteratively.

### 5. `k-means.py`
A Python script for **K-Means clustering**, a machine learning technique to segment and classify image pixels into clusters based on their features.
