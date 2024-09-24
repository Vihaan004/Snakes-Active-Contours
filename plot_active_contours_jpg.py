
# ====================
# Active Contour Model
# ====================
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Load the local image file (replace 'your_image.jpg' with your image path)
img = io.imread('image.jpg')

# Convert the image to grayscale if it is RGB
if len(img.shape) == 3:
    img = rgb2gray(img)

# Initialize the circular contour (you may need to adjust these values based on the image size)
# Approximate values for a face on the left (you can adjust the radius and center as needed)
s = np.linspace(0, 2 * np.pi, 400)
r = 1650 + 400 * np.sin(s)   # Adjust the center and radius (r) for the face on the left
c = 2800 + 400 * np.cos(s)   # Adjust the center and radius (c) for the face on the left
init = np.array([r, c]).T

# Apply the active contour model
snake = active_contour(
    gaussian(img, sigma=3, preserve_range=False),
    init,
    alpha=0.015,
    beta=10,
    gamma=0.001,
)

# Visualize the result
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])

ax.set_ylim(ax.get_ylim()[::-1])

plt.show()
