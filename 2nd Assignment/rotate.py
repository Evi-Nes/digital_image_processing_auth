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
def findRotationAngle(input_image):
    """
    Find the angle of rotation of the image using DFT
    :param input_image: the given image
    :return: The angle for rotation
    """
    connected, thresh = preprocess(input_image)

    # Calculate the DFT of the image and shift the zero-freq component to the center of the spectrum
    f = np.fft.fft2(connected)
    fshift = np.fft.fftshift(f)

    # Calculate the magnitude spectrum of the DFT
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mret, mthresh = cv2.threshold(magnitude_spectrum, 235, 255, cv2.THRESH_BINARY)
    cv2.imwrite("mthresh.jpg", mthresh)

    height, width = mthresh.shape
    polygons = np.array([
        [(0, height/3), (width, height/3.2), (width, 2*height/3), (0, 2*height/3)]  # (y,x)
    ])
    mask = np.zeros_like(mthresh)
    cv2.fillPoly(mask, np.int32([polygons]), 255)
    masked_image = cv2.bitwise_and(mthresh, mask)
    cv2.imshow("masked", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #add 2nd mask
    mask = np.zeros_like(masked_image)
    rows, cols = mask.shape[:2]
    center = (cols // 2, rows // 2)
    radius = 200
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    inverse_mask = 1 - mask / 255  # Invert the mask

    # Multiply the inverse mask with the image using element-wise multiplication
    masked_img = cv2.bitwise_and(masked_image, masked_image, mask=inverse_mask.astype(np.uint8) * 255)

    # Display the result
    cv2.imshow('Masked Image', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Create a copy of the magnitude spectrum and necessary variables
    src = masked_image
    height, width = src.shape
    src = np.array(src, dtype=np.int16)
    dst = np.zeros((height, width), dtype=np.int16)

    slope = np.array([])
    intercept = np.array([])

    # Apply Canny edge detection and HoughLines function
    edges = cv2.Canny(src, dst, 210, 235, 3, False)
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, 20, np.array([]), minLineLength=20, maxLineGap=5)
    lines = lines.squeeze()

    # Draw the lines on the image and calculate the slope and intercept of each line
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(input_image, (x1, y1), (x2, y2), (255, 64, 64), 3)
        if x1 == x2:
            continue
        slope_f = ((y2 - y1) / (x2 - x1))
        intercept_f = (y1 - (slope * x1))

        slope = np.append(slope, slope_f)
        intercept = np.append(intercept, intercept_f)

    cv2.imshow("nl", input_image)
    # Create a boolean mask for the inf values
    mask = np.isinf(slope)

    # Index the array with the inverse of the mask to remove the inf values
    clean_arr = slope[~mask]

    slope = np.mean(clean_arr)
    intercept = np.mean(intercept)
    print("Slope", slope)
    angle_degrees = np.degrees(np.arctan(slope))
    print("Angle", angle_degrees)
    # should be slope=0.4 and angle=20

    return angle_degrees

def serialSearch(input_image, angle_degrees):
    """
    Through a serial search, find the desired angle of rotation of the image
    :param input_image: the given image
    :param angle_degrees: the angle of rotation calculated by findRotationAngle
    :return: the angle of rotation after the serial search
    """
    angle_degrees = 0
    range_degrees = np.arange(np.int32(angle_degrees-10), np.int32(angle_degrees+10), 1)
    variance_normalized_f = np.array([])

    for possible_angle in range_degrees:

        rotated_image = fast_rotate_image(input_image, possible_angle)
        connected, thresh = preprocess(rotated_image)

        # Calculate the DFT of the image and shift the zero-freq component to the center of the spectrum
        f = np.fft.fft2(connected)
        fshift = np.fft.fftshift(f)

        # Calculate the magnitude spectrum of the DFT
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        mret, mthresh = cv2.threshold(magnitude_spectrum, 235, 255, cv2.THRESH_BINARY)
        # display(mthresh)
        vertical_projection = np.sum(mthresh, axis=1)

        # Compute the first derivative of the vertical projection
        d_vertical_projection = np.diff(vertical_projection)

        # Compute the variance of the first derivative
        variance = np.var(d_vertical_projection)

        # Alternatively, you can compute the number of sign changes in the first derivative
        # sign_changes = np.sum(np.abs(np.diff(np.sign(d_vertical_projection))))

        variance_normalized_f = np.append(variance_normalized_f, variance)

    # Normalize the variance/sign_changes to the range [0, 1]
    variance_normalized = variance_normalized_f / np.max(variance_normalized_f)
    # sign_changes_normalized = sign_changes / np.max(sign_changes)

    index = np.argmax(variance_normalized)
    calculated_angle = range_degrees[index]
    print("calculated angle", calculated_angle)
    # weight = 0.8
    #
    # # Calculate the weighted mean
    # weighted_mean = (1 - weight) * angle + weight * calculated_angle
    #
    # # Print the result
    # print("Weighted mean:", weighted_mean)

    serial_angle = calculated_angle

    return serial_angle

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

def rotateImage(input_image, angle_degrees):
    """
    Rotate the image by the given angle
    :param input_image: the given image
    :param angle_degrees: the angle of rotation calculated by serialSearch
    :return: the rotated image
    """

    rows, cols = input_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle_degrees, 1)

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
    image = cv2.imread("image.png")
    angle = findRotationAngle(image)
    serial_angle = serialSearch(image, angle)
    cv2.imwrite("rotated.jpg", rotateImage(image, serial_angle))
