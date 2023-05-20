import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

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
    bw_inverted_image = cv2.bitwise_not(bw_image)

    # connect horizontally oriented regions
    # kernel value (9,1) can be changed to improve the text detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected_image = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE, kernel)
    # display(connected_image)
    inverted_image = cv2.bitwise_not(connected_image)
    # display(inverted_image)
    return inverted_image, bw_inverted_image


def preprocessText(input_image):
    """
    Preprocess the image to make it easier to find the text
    :param input_image: the given image
    :return: the preprocessed image
    """
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # define the kernel and invert the image
    kernel = np.ones((3, 3), np.uint8)
    inverted_image = cv2.bitwise_not(binary_image)

    # dilate the image
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)

    # Remove the dilated image from the original image
    removed_image = cv2.subtract(grayscale, dilated_image)

    # Perform thinning of the result image
    inverted_image = cv2.bitwise_not(removed_image)
    eroded_image = cv2.erode(inverted_image, kernel, iterations=1)

    final_image = np.copy(eroded_image)
    # display(final_image, "final_image")

    return final_image

def detectLines(input_image, display_img):
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)

    # Compute and smooth the vertical projection of brightness
    vertical_projection = cv2.reduce(input_image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
    vertical_projection = cv2.GaussianBlur(vertical_projection, (3, 3), 0)
    row_sum = np.sum(vertical_projection, axis=1)

    # Find the peaks in the vertical projection
    peaks, _ = find_peaks(row_sum, height=900000, distance=20, width=10)
    coordinates = {}

    # Draw the detected lines on the original image
    for i, peak in enumerate(peaks):
        coordinates[i] = peak

        if i == 0:
            continue
        else:
            line = display_img[coordinates[i-1]:peak, 5:display_img.shape[1]-5]
            # Find the rows that contain only white pixels and remove them from the image
            white_rows = np.all(line >= 245, axis=1)
            cropped_image = line[~white_rows, :]

            # cv2.imwrite(f"lines/line{i}.png", cropped_image)

    return coordinates

def detectWords(input_coordinates, input_image, display_img):
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    coords = []

    for i in range(len(input_coordinates)):
        x, y, w, h = 15, input_coordinates[i] - 35, input_image.shape[1]-20, 60
        line = display_img[y:y + h, x:x + w]
        lineb = cv2.blur(line, (25, 25))

        # Compute the horizontal projection of brightness
        horizontal_projection = cv2.reduce(lineb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)

        # Smooth the horizontal projection with a Gaussian filter
        horizontal_projection = cv2.GaussianBlur(horizontal_projection, (5, 5), 0)
        col_sum = np.sum(horizontal_projection, axis=0)

        # Find the peaks in the horizontal projection
        peaks, _ = find_peaks(col_sum, height=15000, distance=60)
        coordinates = []

        # Draw the detected lines on the original image
        for j, peak in enumerate(peaks):
            coordinates.append(peak)

            if j == 0:
                word = line[0:line.shape[0], 15:peak]
            elif j == len(input_coordinates)-1:
                word = line[0:line.shape[0], peak:line.shape[1]-15]
            else:
                word = line[0:line.shape[0], coordinates[j-1]:peak]

            # Save the line image to a file (x,y)
            # cv2.imwrite(f"words/line{i}_word{j + 1}.png", word)

        coords.append(coordinates)

    return coords

def detectLetters(input_coordinates, input_image, display_img):
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    coords = []

    for i in range(len(input_coordinates)):
        if i == 0:
            continue
        # Calculate the coordinates of each line
        x1, y1, x2, y2 = 5, input_coordinates[i-1], display_img.shape[1]-5, input_coordinates[i]
        line = display_img[y1:y2, x1:x2]
        white_rows = np.all(line >= 245, axis=1)
        cropped_image = line[~white_rows, :]

        # Compute and smooth the horizontal projection of brightness
        horizontal_projection = cv2.reduce(cropped_image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        # horizontal_projection = cv2.GaussianBlur(horizontal_projection, (1, 1), 0)
        col_sum = np.sum(horizontal_projection, axis=0)

        # Find the peaks in the horizontal projection
        peaks, _ = find_peaks(col_sum, height=12000, distance=25, width=5)
        coordinates = []
        lcoordinates = []

        # Unpack the coordinates of each letter
        for j, peak in enumerate(peaks):
            coordinates.append(peak)

            if j == 0:
                continue
            else:
                letter = cropped_image[0:cropped_image.shape[0], coordinates[j-1]:peak]
                lcoords = (x1 + coordinates[j-1], y1, x1 + peak, y1)

            # Save each letter to a file
            cv2.imwrite(f"letters/line{i}_letter{j}.png", letter)

            lcoordinates.append(lcoords)

        coords.append(lcoordinates)

    return coords

def returnCharacters(filepath):
    chars = []  # List to store all characters

    with open(filepath, 'r') as file:
        for line in file:
            for char in line:
                if char == ' ' or char == '\n':
                    continue
                chars.append(char)

    return chars


if __name__ == "__main__":
    image = cv2.imread("text1_v2.png")
    display_image = np.copy(image)
    connected, thresh = preprocessImage(image)

    lines_coordinates = detectLines(connected, display_image)
    # words_coordinates = detectWords(lines_coordinates, thresh, display_image)
    letter_coordinates = detectLetters(lines_coordinates, thresh, display_image)

    # proccessed_image = preprocessText(display_image)
    # file_path = 'text1_v2.txt'  # Replace with the actual path to your text file
    # characters = returnCharacters(file_path)
    #
    # y = characters
    # X = []
    #
    # for line in letter_coordinates:
    #     for letter in line:
    #         x1, y1, x2, y2 = letter
    #         X.append(proccessed_image[y1:y2, x1:x2])
    #         # breakpoint()
    #
    # for i in range(len(X)):
    #     X[i] = cv2.resize(X[i], (70, 70))
    # # max_length = max(len(sublist) for sublist in X)
    # # X = [sublist + [0] * (max_length - len(sublist)) for sublist in X]
    #
    # # Convert to numpy arrays
    # X = np.array(X)
    # y = np.array(y)
    #
    # # Step 3: Split the dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #
    # # Step 4: Normalize the features
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # # Step 5: Train the KNN model
    # k = 3  # Number of neighbors
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X_train, y_train)
    #
    # # Step 6: Make predictions
    # y_pred = knn.predict(X_test)
    #
    # # Step 7: Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")
    #
    # # Step 10: Predict on new, unseen data
    # new_image_paths = ['dataset/g.png', 'dataset/b.png']
    # new_data = []
    #
    # for path in new_image_paths:
    #     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #     _, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #     contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contour_count = len(contours)
    #
    #     new_data.append(contour_count)
    #
    # new_data = np.array(new_data).reshape(-1, 1)
    # new_data = scaler.transform(new_data)
    # predicted_classes = knn.predict(new_data)
    # print(f"Predicted classes for new data: {predicted_classes}")