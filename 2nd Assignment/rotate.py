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
    # cv2.imwrite("magnitude.jpg", mthresh)
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
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        # Compute intersection points of line with image boundaries
        x_left = 0
        y_left = int((x_left - x0) / a + y0)
        x_right = image.shape[1] - 1
        y_right = int((x_right - x0) / a + y0)
        y_top = 0
        x_top = int((y_top - y0) / b + x0)
        y_bottom = image.shape[0] - 1
        x_bottom = int((y_bottom - y0) / b + x0)

        # Check if intersection points are within bounds of image
        intersection_points = []
        if y_left >= 0 and y_left < image.shape[0]:
            intersection_points.append((x_left, y_left))
        if y_right >= 0 and y_right < image.shape[0]:
            intersection_points.append((x_right, y_right))
        if x_top >= 0 and x_top < image.shape[1]:
            intersection_points.append((x_top, y_top))
        if x_bottom >= 0 and x_bottom < image.shape[1]:
            intersection_points.append((x_bottom, y_bottom))

        # If intersection points are not within bounds of image, compute intersection points with extended lines
        if len(intersection_points) < 2:
            if y_left < 0:
                x_left = int((y_left - y0) / b + x0)
                intersection_points.append((x_left, 0))
            elif y_left >= image.shape[0]:
                x_left = int((y_left - y0) / b + x0)
                intersection_points.append((x_left, image.shape[0] - 1))
            if y_right < 0:
                x_right = int((y_right - y0) / b + x0)
                intersection_points.append((x_right, 0))
            elif y_right >= image.shape[0]:
                x_right = int((y_right - y0) / b + x0)
                intersection_points.append((x_right, image.shape[0] - 1))
            if x_top < 0:
                y_top = int((x_top - x0) / a + y0)
                intersection_points.append((0, y_top))
            elif x_top >= image.shape[1]:
                y_top = int((x_top - x0) / a + y0)
                intersection_points.append((image.shape[1] - 1, y_top))
            if x_bottom < 0:
                y_bottom = int((x_bottom - x0) / a + y0)
                intersection_points.append((0, y_bottom))
            elif x_bottom >= image.shape[1]:
                y_bottom = int((x_bottom - x0) / a + y0)
                intersection_points.append((image.shape[1] - 1, y_bottom))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # if in comment, the code will work correctly for text1.png but calculates wrong angle for image.png
        # if not in comment, the code will work correctly for image.png but calculates wrong angle for text1.png
        # if len(intersection_points) >= 2:
        #     x1, y1 = intersection_points[0]
        #     x2, y2 = intersection_points[1]

        slope_f = ((y2 - y1) / (x2 - x1))
        intercept_f = (y2 - slope_f * x2)

        slope = np.append(slope, slope_f)
        intercept = np.append(intercept, intercept_f)

    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    slope = np.mean(slope)
    intercept = np.mean(intercept)
    print("Slope", slope)
    angle_degrees = np.degrees(np.arctan(slope))
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
