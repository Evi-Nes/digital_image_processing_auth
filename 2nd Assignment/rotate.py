import cv2
import numpy as np
import matplotlib.pyplot as plt

# def findRotationAngle(image):
#     """
#     Find the angle of rotation of the image
#     :param image: the given image
#     :return: The angle for rotation
#     """
#     angle = 10
#     return angle

def rotateImage(image):

    image = cv2.blur(image, (15, 15))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image to convert it to a binary image
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate the DFT of the binary image using Numpy's FFT
    f = np.fft.fft2(thresh)

    # Shift the zero-frequency component to the center of the spectrum using Numpy's fftshift
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum of the DFT
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    cv2.imwrite("magnit.jpg", magnitude_spectrum)

    # Find the location of the maximum value in the magnitude spectrum using Numpy's argmax
    rows, cols = thresh.shape[:2]
    crow, ccol = int(rows / 2), int(cols / 2)
    max_value_location = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)

    # Calculate the angle of rotation based on the location of the maximum value in the magnitude spectrum
    angle = np.arctan2(max_value_location[0] - crow, max_value_location[1] - ccol)
    angle_degrees = np.degrees(angle)

    # Rotate the image by the calculated angle to fine-tune the rotation
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle_degrees, 1)
    # Calculate new image dimensions
    cos_theta = abs(M[0, 0])
    sin_theta = abs(M[0, 1])
    new_width = int((rows * sin_theta) + (cols * cos_theta))
    new_height = int((rows * cos_theta) + (cols * sin_theta))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - cols // 2
    M[1, 2] += (new_height / 2) - rows // 2
    rotated_img = cv2.warpAffine(image, M, (new_width, new_height))

    return rotated_img

if __name__ == "__main__":
    image = cv2.imread("image.png")
    # angle = findRotationAngle(image)
    cv2.imwrite("rotated.jpg", rotateImage(image))
