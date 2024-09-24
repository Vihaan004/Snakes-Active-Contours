# ====================
# Active Contour Model
# ====================
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray  # Converts RGB images to grayscale
from skimage import io              # For reading images from files
from skimage.filters import gaussian  # Applies Gaussian smoothing to an image
from skimage.segmentation import active_contour  # Active contour (snakes) algorithm

# Load the local image file (replace 'cube.jpg' with your image file)
# This function reads the image into a NumPy array.
img = io.imread('cube.jpg')

# Convert the image to grayscale if it is in RGB format (i.e., it has 3 channels).
# This is because the active contour model works better with single-channel (grayscale) images.
if len(img.shape) == 3:
    img = rgb2gray(img)

# Initialize the circular contour
# We are generating a parametric circular contour using sine and cosine functions.
# 's' represents angles in radians (0 to 2*pi), and we use sine and cosine to create a circle.
# 'r' represents the row (y-coordinate) values of the circle.
# 'c' represents the column (x-coordinate) values of the circle.
# These are based on the center of the circle (100, 100) and a radius of 100 pixels.
s = np.linspace(0, 2 * np.pi, 400)  # 400 points evenly spaced from 0 to 2*pi
r = 100 + 100 * np.sin(s)   # Y-coordinates (rows) of the circular contour
c = 100 + 100 * np.cos(s)   # X-coordinates (columns) of the circular contour
init = np.array([r, c]).T   # Create a 2D array with the contour coordinates

# Apply the active contour model (snakes algorithm)
# This function evolves the initial contour to fit around image features, like edges or lines.
# 'gaussian' applies a Gaussian filter (blurs the image) to reduce noise and avoid small edges.
# 'alpha' controls the elasticity of the contour (how much it can stretch).
# 'beta' controls the rigidity of the contour (how much it resists bending).
# 'gamma' is a time step for the optimization process (smaller values mean slower but more precise convergence).
snake = active_contour(
    gaussian(img, sigma=3, preserve_range=False),  # Apply Gaussian smoothing to the image
    init,  # Initial circular contour to evolve
    alpha=0.015,  # Elasticity of the contour
    beta=10,  # Rigidity of the contour
    gamma=0.001,  # Time step for the evolution process
)

# Visualize the result using Matplotlib
# Plot the original image in grayscale, the initial contour in red, and the final contour in blue
fig, ax = plt.subplots(figsize=(7, 7))  # Create a figure and axes for plotting
ax.imshow(img, cmap=plt.cm.gray)  # Display the image in grayscale
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)  # Plot the initial contour (in red dashed line)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)  # Plot the final contour after snake optimization (in blue)
ax.set_xticks([]), ax.set_yticks([])  # Remove axis ticks for a cleaner image display
ax.axis([0, img.shape[1], img.shape[0], 0])  # Set axis limits to fit the image dimensions

plt.show()  # Display the plot
