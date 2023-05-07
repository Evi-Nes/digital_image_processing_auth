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

def masked_region(image, height, width):
    # mask over the whole frame
    polygons = np.array([
        [(0, 0), (width, 0), (width, height/2), (0, height/2)]  # (y,x)
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32([polygons]),255)
    masked_image = cv2.bitwise_and(image, mask)
    cv2.imshow("masked", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return masked_image
def rotateImage(image):

    # Pre-process the image and convert it to a binary image
    image = cv2.blur(image, (15, 15))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape

    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate the DFT of the image and shift the zero-freq component to the center of the spectrum using np.fftshift
    f = np.fft.fft2(thresh)
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum of the DFT
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    masked_image = masked_region(magnitude_spectrum, height, width)
    mret, mthresh = cv2.threshold(masked_image, 240, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("mthresh.jpg", mthresh)
    cv2.imshow("mthresh", mthresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply Canny edge detection on the magnitude spectrum
    src = magnitude_spectrum
    height, width = src.shape
    src = np.array(src, dtype=np.int16)
    dst = np.zeros((height, width), dtype=np.int16)

    edges = cv2.Canny(masked_image, dst, 210, 235, 3, False)

    # Apply HoughLines function
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 20, np.array([]), minLineLength=5, maxLineGap=5)
    lines = lines.squeeze()

    slope = np.array([])
    intercept = np.array([])

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (255, 64, 64), 3)
        # if np.abs(y2 - y1) < 10:
        #     continue
        # else:
        slope_f = ((y2 - y1) / (x2 - x1))
        intercept_f = (y1 - (slope * x1))

        slope = np.append(slope, slope_f)
        intercept = np.append(intercept, intercept_f)

    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    slope = np.mean(slope)
    intercept = np.mean(intercept)
    print("Slope", slope)
    angle_degrees = -np.degrees(np.arctan(slope))
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
