import cv2
import numpy as np

debug = True

# Display image
def display(input_image, frame_name="OpenCV Image"):
    if not debug:
        return
    h, w = input_image.shape[0:2]
    new_w = 800
    new_h = int(new_w * (h / w))
    input_image = cv2.resize(input_image, (new_w, new_h))
    cv2.imshow(frame_name, input_image)
    cv2.waitKey(0)

def preprocess(input_image):
    """
    Preprocess the image to get the text regions
    :param input_image: the given image
    :return: connected_image the image with connected text regions
    bw_image: the binarized image
    """
    small = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    # display(grad)

    # Binarize the gradient image
    _, bw_image = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # display(bw_image)

    # connect horizontally oriented regions
    # kernel value (9,1) can be changed to improve the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected_image = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE, kernel)
    # display(connected_image)

    return connected_image, bw_image
def findRotationAngle(input_image, disp_image):
    """
    Find the angle of rotation of the image using DFT and magnitude spectrum
    :param disp_image: copy of the original image
    :param input_image: the preprocessed image
    :return: the angle for rotation
    """
    height, width = input_image.shape[:2]

    # Calculate the DFT of the image and shift the zero-freq component to the center of the spectrum
    f = np.fft.fft2(input_image)
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum of the DFT
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mret, mthresh = cv2.threshold(magnitude_spectrum, 235, 255, cv2.THRESH_BINARY)
    display(mthresh)

    # Create a copy of the magnitude spectrum and necessary variables
    src = mthresh
    src = np.array(src, dtype=np.int16)
    dst = np.zeros((height, width), dtype=np.int16)

    # Apply Canny edge detection and HoughLines function
    edges = cv2.Canny(src, dst, 210, 235, 3, False)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)

    # Define size threshold and remove small objects
    min_size = 10
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            edges[labels == i] = 0

    display(edges)

    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 20, np.array([]), minLineLength=20, maxLineGap=5)
    lines = lines.squeeze()

    center = (width // 2, height // 2)
    radius = 110
    slope = np.array([])

    # Calculate the slope of each line and draw the lines on the image
    for line in lines:
        x1, y1, x2, y2 = line

        if (y1 < (center[1] - radius)) | (y2 < (center[1] - radius)) | (y1 > (center[1] + radius)) | (y2 > (center[1] + radius)):
            if x1 == x2:
                x1 = x1 + 1

            slope = np.append(slope, ((y2 - y1) / (x2 - x1)))
            cv2.line(disp_image, (x1, y1), (x2, y2), (255, 64, 64), 3)

        else:
            continue

    display(disp_image)

    slope = np.mean(slope)
    angle_degrees = -(90 - np.degrees(np.arctan(slope)))
    print("Angle", angle_degrees)

    return angle_degrees

def serialSearch(input_image, angle_degrees):
    """
    Through a serial search, find the desired angle of rotation of the image
    :param input_image: the given image
    :param angle_degrees: the angle of rotation calculated by findRotationAngle
    :return: the angle of rotation after the serial search
    """
    range_degrees = np.arange(np.int32(angle_degrees-10), np.int32(angle_degrees+10), 1)
    variance_normalized_f = np.array([])

    for possible_angle in range_degrees:
        rotated_image = fast_rotate_image(input_image, possible_angle)

        # Calculate the DFT of the image and shift the zero-freq component to the center of the spectrum
        f = np.fft.fft2(rotated_image)
        fshift = np.fft.fftshift(f)

        # Calculate the magnitude spectrum of the DFT
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        mret, mthresh = cv2.threshold(magnitude_spectrum, 235, 255, cv2.THRESH_BINARY)
        vertical_projection = np.sum(mthresh, axis=1)

        # Compute the first derivative of the vertical projection
        d_vertical_projection = np.diff(vertical_projection)

        # Compute the variance of the first derivative
        variance_normalized_f = np.append(variance_normalized_f, np.var(d_vertical_projection))

    # Normalize the variance/sign_changes to the range [0, 1]
    variance_normalized = variance_normalized_f / np.max(variance_normalized_f)

    index = np.argmax(variance_normalized)
    calculated_angle = range_degrees[index]
    print("serial angle", calculated_angle)

    final_angle = np.int32((calculated_angle + angle_degrees)/2)
    print("final angle", final_angle)

    return final_angle

def fast_rotate_image(input_image, input_angle):
    height, width = input_image.shape[:2]
    image_center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(image_center, input_angle, 1.)

    # Calculate new image dimensions
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
    rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

    # Perform the actual rotation and pad the unused area with black
    result = cv2.warpAffine(input_image, rotation_matrix, (new_width, new_height))

    return result

def rotateImage(input_image, rotation_angle):
    """
    Rotate the image by the given angle
    :param input_image: the given image
    :param rotation_angle: the angle of rotation calculated by serialSearch
    :return: the rotated image
    """

    rows, cols = input_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotation_angle, 1)

    # Calculate new image dimensions
    cos_theta = abs(M[0, 0])
    sin_theta = abs(M[0, 1])
    new_width = int((rows * sin_theta) + (cols * cos_theta))
    new_height = int((rows * cos_theta) + (cols * sin_theta))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_width / 2) - cols // 2
    M[1, 2] += (new_height / 2) - rows // 2
    rotated_img = cv2.warpAffine(input_image, M, (new_width, new_height))

    return rotated_img


if __name__ == "__main__":
    image = cv2.imread("image2.png")
    display_image = np.copy(image)
    connected, thresh = preprocess(image)

    angle = findRotationAngle(connected, display_image)
    serial_angle = serialSearch(connected, angle)

    cv2.imwrite("rotated_image.jpg", rotateImage(image, serial_angle))
