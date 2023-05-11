import cv2
import numpy as np

debug = True

def display(input_image, frame_name="OpenCV Image"):
    if not debug:
        return
    h, w = input_image.shape[0:2]
    new_w = 800
    new_h = int(new_w * (h / w))
    input_image = cv2.resize(input_image, (new_w, new_h))
    cv2.imshow(frame_name, input_image)
    cv2.waitKey(0)

def preprocessImage(input_image):
    """
    Preprocess the image to get the text regions
    :param input_image: the given image
    :return: connected_image the image with connected text regions
    bw_image: the binarized image
    """
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # find the gradient map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(grayscale, cv2.MORPH_GRADIENT, kernel)

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

def detectLines(input_image, display_image):
    # Find the contours in the resulting image
    contours, hierarchy = cv2.findContours(input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours from up to bottom based on their x-coordinate values
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    coordinates = {}
    # Extract each line of text from the original image

    # Merge contours that are close enough together
    merged_contours = []
    current_contour = contours[0]

    for contour in contours[1:]:
        # Check if the current contour is close enough to the next contour
        x1, y1, w1, h1 = cv2.boundingRect(current_contour)
        x2, y2, w2, h2 = cv2.boundingRect(contour)
        if y2 -y1 < h1 / 2:
            # Merge the contours into a single contour
            current_contour = cv2.convexHull(np.concatenate((current_contour, contour)))
        else:
            # Add the current contour to the list of merged contours
            merged_contours.append(current_contour)
            current_contour = contour

    # Add the last contour to the list of merged contours
    merged_contours.append(current_contour)

    for i, contour in enumerate(merged_contours):
        # Extract the bounding box coordinates for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the line of text from the original image
        line = display_image[y:y + h, x:x + w]
        coordinates[i] = (x, y, w, h)
        # Save the line image to a file
        cv2.imwrite(f"lines/line{i + 1}.png", line)
    return coordinates

def detectWords(lcoordinates, thresh, display_image):
    for i in range(len(lcoordinates)):
        x, y, w, h = lcoordinates[i]
        # Extract the line of text from the original image
        line = thresh[y:y + h, x:x + w]
        line = cv2.blur(line, (4, 4))
        # display(line)
        # Find the contours in the line image
        contours, hierarchy = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours from left to right based on their x-coordinate values
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        coordinates = {}
        # Iterate through each contour and compute the bounding rectangle
        for j, contour in enumerate(contours):
            # Extract the bounding box coordinates for the contour
            x2, y2, w2, h2 = cv2.boundingRect(contour)
            coordinates[i] = (x2, y2, w2, h2)
            # Extract the word from the original image
            word = display_image[y + y2:y + y2 + h2, x + x2:x + x2 + w2]
            # display_image(word)
            # Save the word image to a file
            cv2.imwrite(f"words/line{i + 1}_word{j + 1}.png", word)
    return coordinates

def detectLetters(wcoordinates, thresh, display_image):
    for i in range(len(wcoordinates)):
        for j in range(len(wcoordinates[i])):
            x, y, w, h = wcoordinates[i][j]
            # Extract the line of text from the original image
            word = thresh[y:y + h, x:x + w]
            word = cv2.blur(word, (4, 4))
            # display(word)
            # Find the contours in the line image
            contours, hierarchy = cv2.findContours(word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort the contours from left to right based on their x-coordinate values
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            coordinates = {}
            # Iterate through each contour and compute the bounding rectangle
            for k, contour in enumerate(contours):
                # Extract the bounding box coordinates for the contour
                x2, y2, w2, h2 = cv2.boundingRect(contour)
                coordinates[i][j][k] = (x2, y2, w2, h2)
                # Extract the word from the original image
                letter = display_image[y + y2:y + y2 + h2, x + x2:x + x2 + w2]
                # display_image(word)
                # Save the word image to a file
                cv2.imwrite(f"letters/line{i + 1}_word{j + 1}_letter{k + 1}.png", letter)
    return coordinates

if __name__ == "__main__":
    image = cv2.imread("text1.png")
    display_image = np.copy(image)
    connected, thresh = preprocessImage(image)
    coordinates ={}
    wcoordinates = detectLines(connected, display_image)
    lcoordinates = detectWords(wcoordinates, thresh, display_image)
    detectLetters(lcoordinates, thresh, display_image)


