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
    # cv2.imshow("mthresh", mthresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    src = magnitude_spectrum
    height, width = src.shape
    dst = np.zeros((height, width), dtype=np.int16)
    src = np.array(src, dtype=np.int16)

    edges = cv2.Canny(src, dst, 190, 220, 3, False)

    slope = np.array([])
    intercept = np.array([])
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
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        slope_f = ((y2 - y1) / (x2 - x1))
        intercept_f = (y2 - slope * x2)

        slope = np.append(slope, slope_f)
        intercept = np.append(intercept, intercept_f)

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    slope = np.average(slope)
    intercept = np.average(intercept)
    print("Slope", slope)
    angle_degrees = - np.degrees(np.arctan(slope))
    print("Angle", angle_degrees)
    # should be slope=0.4 and angle=20

    rows, cols = thresh.shape[:2]

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
