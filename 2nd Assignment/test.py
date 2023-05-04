import cv2
import numpy as np

# Load the images
img1 = cv2.imread('text1.png', 0)
img2 = cv2.imread('image.png', 0)

img1 = cv2.blur(img1, (15, 15))

img2 = cv2.blur(img2, (15, 15))


# Define the template size
template_size = (50, 50)

# Define the search range
search_range = (img1.shape[1] - template_size[1], img1.shape[0] - template_size[0])

# Create a template from the middle of the first image
template = img1[int(img1.shape[0] / 2 - template_size[0] / 2) : int(img1.shape[0] / 2 + template_size[0] / 2),
                int(img1.shape[1] / 2 - template_size[1] / 2) : int(img1.shape[1] / 2 + template_size[1] / 2)]

# Perform template matching to find the best match in the second image
result = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Calculate the translation between the two images
tx = max_loc[0] - img2.shape[1] / 2
ty = max_loc[1] - img2.shape[0] / 2

# Apply the translation to the second image to align it with the first image
M = np.float32([[1, 0, tx], [0, 1, ty]])
aligned_img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

# Get the shape of aligned_img2
h, w = aligned_img2.shape[:2]

# Resize img1 to match the shape of aligned_img2
img1_resized = cv2.resize(img1, (w, h))

# Show the aligned images side by side
aligned = np.concatenate((img1_resized, aligned_img2), axis=1)
cv2.imshow('Aligned Images', aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
