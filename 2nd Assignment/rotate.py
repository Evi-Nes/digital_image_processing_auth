import cv2
import numpy as np

def findRotationAngle(image):
    """
    Find the angle of rotation of the image
    :param image: the given image
    :return: The angle for rotation
    """
    # Pre-process the image and convert it to a binary image
    image = cv2.blur(image, (15, 15))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate the DFT of the image and shift the zero-freq component to the center of the spectrum using np.fftshift
    f = np.fft.fft2(thresh)
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum of the DFT
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mret, mthresh = cv2.threshold(magnitude_spectrum, 240, 255, cv2.THRESH_BINARY)
    cv2.imwrite("mthresh.jpg", mthresh)

    # Create a copy of the magnitude spectrum and necessary variables
    src = magnitude_spectrum
    height, width = src.shape
    src = np.array(src, dtype=np.int16)
    dst = np.zeros((height, width), dtype=np.int16)

    slope = np.array([])
    intercept = np.array([])

    # Apply Canny edge detection and HoughLines function
    edges = cv2.Canny(src, dst, 210, 235, 3, False)
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 20, np.array([]), minLineLength=5, maxLineGap=5)
    lines = lines.squeeze()

    # Draw the lines on the image and calculate the slope and intercept of each line
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (255, 64, 64), 3)

        slope_f = ((y2 - y1) / (x2 - x1))
        intercept_f = (y1 - (slope * x1))

        if slope_f > 1.8 or slope_f < -1.8:
            continue
        else:
            slope = np.append(slope, slope_f)
            intercept = np.append(intercept, intercept_f)

    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate the average slope and intercept and the angle of rotation
    slope = np.mean(slope)
    intercept = np.mean(intercept)
    print("Slope", slope)
    angle_degrees = np.degrees(np.arctan(slope))
    print("Angle", angle_degrees)
    # should be slope=0.4 and angle=20

    return angle_degrees

def serialSearch(image, angle_degrees):
    """
    Through a serial search, find the desired angle of rotation of the image
    :param image: the given image
    :param angle_degrees: the angle of rotation calculated by findRotationAngle
    :return: the angle of rotation
    """

    return serial_angle

def rotateImage(image, angle_degrees):
    """
    Rotate the image by the given angle
    :param image: the given image
    :param angle_degrees: the angle of rotation calculated by serialSearch
    :return: the rotated image
    """

    rows, cols = image.shape[:2]
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
    angle = findRotationAngle(image)
    serial_angle = serialSearch(image, angle)
    cv2.imwrite("rotated.jpg", rotateImage(image, serial_angle))
