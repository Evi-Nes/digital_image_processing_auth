import cv2
import numpy as np
from scipy.signal import find_peaks

# Load the image and convert to grayscale
img = cv2.imread('text1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to create a binary image
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Perform DFT to analyze the frequency components
dft = cv2.dft(np.float32(thresh), flags=cv2.DFT_COMPLEX_OUTPUT)
magnitude = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
magnitude = cv2.log(1 + magnitude)

# Compute the vertical projection of brightness
vertical_projection = np.sum(thresh, axis=1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vertical_projection = cv2.reduce(img_gray, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)

# Smooth the vertical projection with a Gaussian filter
vertical_projection = cv2.GaussianBlur(vertical_projection, (5, 5), 0)

# Find the peaks in the vertical projection
row_sum = np.sum(vertical_projection, axis=1)
row_sum = row_sum.astype(np.float32)

peaks, _ = find_peaks(row_sum, height=100, distance=20)
coordinates = {}

# Draw the detected lines on the original image
for i, peak in enumerate(peaks):
    coordinates[i] = peak
    cv2.line(img, (0, peak), (img.shape[1], peak), (0, 0, 255), thickness=2)

# Display the image with detected lines
cv2.imshow('Detected Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(len(coordinates)):
    if i == 0:
        line = img[0:coordinates[i], 0:img.shape[1]]
    elif i == (len(coordinates) - 1):
        line = img[coordinates[i]:coordinates[i]+25, 0:img.shape[1]]
    else:
        line = img[coordinates[i-1]:coordinates[i], 0:img.shape[1]]

    # Save the line image to a file
    cv2.imwrite(f"lines/line{i + 1}.png", line)

