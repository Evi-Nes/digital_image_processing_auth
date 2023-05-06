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
    mret, mthresh = cv2.threshold(magnitude_spectrum, 240, 255, cv2.THRESH_BINARY)
    cv2.imwrite("magnit.jpg", mthresh)

    # # Find the maximum value in each row to obtain a 1D array
    # max_vals = np.max(magnitude_spectrum, axis=1)
    #
    # # Fit a line to the 1D array using polyfit
    # y = np.arange(len(max_vals))
    # coeffs = np.polyfit(y, max_vals, 1)
    # slope = coeffs[0]
    #
    # # Compute the angle of the line relative to the vertical direction
    # angle_degrees = np.degrees(np.arctan(slope))
    # print(angle_degrees)

    # # Find the location of the maximum value in the magnitude spectrum using Numpy's argmax
    # rows, cols = thresh.shape[:2]
    # crow, ccol = int(rows / 2), int(cols / 2)
    # max_value_location = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
    #
    # # Calculate the angle of rotation based on the location of the maximum value in the magnitude spectrum
    # angle = np.arctan2(max_value_location[0] - crow, max_value_location[1] - ccol)
    # angle_degrees = np.degrees(angle)
    #
    # # Rotate the image by the calculated angle to fine-tune the rotation
    # M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle_degrees, 1)
    # # Calculate new image dimensions
    # cos_theta = abs(M[0, 0])
    # sin_theta = abs(M[0, 1])
    # new_width = int((rows * sin_theta) + (cols * cos_theta))
    # new_height = int((rows * cos_theta) + (cols * sin_theta))
    #
    # # Adjust the rotation matrix to take into account translation
    # M[0, 2] += (new_width / 2) - cols // 2
    # M[1, 2] += (new_height / 2) - rows // 2
    # rotated_img = cv2.warpAffine(image, M, (new_width, new_height))

    src = magnitude_spectrum
    height, width = src.shape
    dst = np.zeros((height, width), dtype=np.int16)
    src = np.array(src, dtype=np.int16)

    edges = cv2.Canny(src, dst, 190, 220, 3, False)
    cv2.imshow('Result', edges)
    cv2.waitKey(0)
    slope = np.zeros((2, 1), dtype=np.int16)
    intercept = np.zeros((2, 1), dtype=np.int16)
    # Apply HoughLines function
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    lines = lines.squeeze()
    # Draw the detected lines on the original image
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        slope_f = (y2 - y1) / (x2 - x1)
        intercept_f = y1 - slope * x1

        slope = np.append(slope, slope_f)
        intercept = np.append(intercept, intercept_f)

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Result', image)

    slope = np.average(slope)
    intercept = np.average(intercept)
    print(slope)

    # Define starting and ending points of the line
    x1 = 0
    y1 = int(slope * x1 + intercept)
    x2 = image.shape[1]
    y2 = int(slope * x2 + intercept)

    # Draw line on the image
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Show the image
    cv2.imshow("Image with line", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return rotated_img


if __name__ == "__main__":
    image = cv2.imread("image.png")
    # angle = findRotationAngle(image)
    cv2.imwrite("rotated.jpg", rotateImage(image))
