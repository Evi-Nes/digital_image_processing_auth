import numpy as np
import cv2

# Make sure to set this to True when you want to run all the functions and detect the corners
debug = False

def myLocalDescriptor(img, p, r_min, r_max, r_step, num_points):
    """
    Computes the local descriptor for each pixel in the image, using circles of different radius.
    :param img: the given image
    :param p: the given pixel
    :param r_min: the minimum radius
    :param r_max: the maximum radius
    :param r_step: the step of the radius
    :param num_points: the number of points in each circle
    :return: descriptor that contains a value for each radius
    """
    size = (r_max - r_min) // r_step
    d = np.full((1, size), 1e20)

    if p[0] + r_max > img.shape[1] or p[1] + r_max > img.shape[0] or p[0] - r_max < 0 or p[1] - r_max < 0:
        return d

    index = 0
    for radius in range(r_min, r_max, r_step):
        x_rho = []
        for theta in range(0, 360, 360 // num_points):
            x = int(p[0] + radius * np.cos(theta))
            y = int(p[1] + radius * np.sin(theta))
            x_rho.append(img[y, x])

        d[0, index] = np.mean(x_rho)
        index += 1
    return d
def myDetectHarrisFeatures(display_img, gray_img):
    """
    Detects all the corners in the given image using the derivatives of x-axis and y-axis.
    :param gray_img: the given grayscale image
    :param display_img: the given image used for cv2.circle
    :return: the detected corners [x,y]
    """
    img_gaussian = cv2.bilateralFilter(gray_img, 11, 80, 80)
    k = 0.04
    r_thresh = 0.30
    offset = 8
    height = gray_img.shape[0]
    width = gray_img.shape[1]
    matrix_R = np.zeros((height, width))
    cornerList = []

    # Calculate the x and y image derivatives
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate product and second derivatives
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    # Calculate derivatives per pixel
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(dx2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            # Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            R = np.linalg.det(H) - k * (np.matrix.trace(H) ** 2)
            matrix_R[y - offset, x - offset] = R

    # Normalize the R values in the range [0, 1]
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if matrix_R[y, x] > r_thresh:
                cornerList.append([x, y])
                cv2.circle(display_img, (x, y), 1, (0, 255, 0), 1)

    cv2.imwrite("my_corners_img.jpg", display_img)

    return cornerList
def filterClosePoints(coords, distance_threshold):
    """
    Removes the points that are closer than the distance_threshold with each other
    :param coords: the coordinates of the detected corners
    :param distance_threshold: the minimum distance between two points
    :return: a list that contains the filtered coordinates
    """
    filtered_coordinates = []

    for i, (x1, y1) in enumerate(coords):
        is_close = False

        for j, (x2, y2) in enumerate(coords[i + 1:], start=i + 1):
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            if distance < distance_threshold:
                is_close = True
                break

        if not is_close:
            filtered_coordinates.append((x1, y1))

    return filtered_coordinates
def preProcessCorners(img, gray, r_min, r_max, r_step, num_per_circle, matrix_size):
    """
    Detects the corners of the image and filters the close points.
    :param img: the given image
    :param gray: the given grayscale image
    :param r_min, r_max, r_step, num_per_circle, matrix_size: the parameters for the local descriptor
    :return: the coordinates of the filtered corners and the descriptor for each corner
    """
    coordinates = myDetectHarrisFeatures(img, gray)
    print('coords', len(coordinates))

    # filtered_coordinates = filterClosePoints(coordinates, distance_threshold=5)
    # print('filtered', len(filtered_coordinates))

    descriptor = np.zeros((len(coordinates), matrix_size))
    for i, point in enumerate(coordinates):
        descriptor[i, :] = myLocalDescriptor(gray, point, r_min, r_max, r_step, num_per_circle)

    return coordinates, descriptor

def calculateDistances(corners1, corners2, descriptors1, descriptors2):
    """
    Calculates the Euclidean distances as the absolute value of the difference between the two points.
    :param corners1: the detected corners from the first image
    :param corners2: the detected corners from the second image
    :return: the Euclidean distances
    """
    distances = np.zeros((len(corners1), len(corners2)))
    for index1, corner1 in enumerate(corners1):
        for index2, corner2 in enumerate(corners2):
            descriptor1 = np.array(descriptors1[index1])
            descriptor2 = np.array(descriptors2[index2])
            distances[index1, index2] = np.abs(np.linalg.norm(descriptor1 - descriptor2))

    np.save("distances.npy", distances)
def descriptorMatching(p1, p2, thresh):
    """
    Matches the descriptors of two points of the two images and returns the 30% of the matched points
    :param p1: the dictionary of first image
    :param p2: the dictionary of second image
    :param thresh: the percentage of the matched points we want to return
    :return: a list that contains the matched points
    """
    corners1, descriptors1 = p1["corners"], p1["descriptor"]
    corners2, descriptors2 = p2["corners"], p2["descriptor"]

    if debug:
        calculateDistances(corners1, corners2, descriptors1, descriptors2)

    distances = np.load("distances.npy")
    matched_points = []

    for index, corner in enumerate(corners1):
        distance = distances[index]
        sorted_distance = np.argsort(distance)
        filtered_distance = sorted_distance[:int(thresh * len(sorted_distance))]
        for filterd in filtered_distance:
            matched_points.append((index, filterd))

    return matched_points

def myDescriptorMatching(img1, img2, thresh, image1, image2):
    img1 = np.load('img1.npy', allow_pickle=True).item()
    keypoint1, descriptors1 = img1["corners"], img1["descriptor"]
    img2 = np.load('img2.npy', allow_pickle=True).item()
    keypoint2, descriptors2 = img2["corners"], img2["descriptor"]

    descriptors1 = np.array(descriptors1, dtype=np.float32)
    descriptors2 = np.array(descriptors2, dtype=np.float32)
    keypoints1_updated = []
    keypoints2_updated = []

    for x, y in keypoint1:
        keypoint = cv2.KeyPoint(x, y, 1)
        keypoints1_updated.append(keypoint)

    for x, y in keypoint2:
        keypoint = cv2.KeyPoint(x, y, 1)
        keypoints2_updated.append(keypoint)

    # finding the nearest match with KNN algorithm
    index_params = dict(algorithm=0, trees=20)
    search_params = dict(checks=150)  # or pass empty dictionary

    # Initialize the FlannBasedMatcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    Matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Need to draw only good matches, so create a mask
    good_matches = [[0, 0] for i in range(len(Matches))]
    # Ratio Test
    for i, (m, n) in enumerate(Matches):
        if m.distance < 0.55 * n.distance:
            good_matches[i] = [1, 0]

    # Extract the matched keypoints' coordinates
    matched_points1 = []
    matched_points2 = []

    for i, match in enumerate(Matches):
        if good_matches[i][0] == 1:
            idx1 = match[0].queryIdx
            idx2 = match[0].trainIdx
            pt1 = keypoints1_updated[idx1].pt
            pt2 = keypoints2_updated[idx2].pt
            matched_points1.append(pt1)
            matched_points2.append(pt2)

    # Draw the matches using drawMatchesKnn()
    Matched = cv2.drawMatchesKnn(image1, keypoints1_updated, image2, keypoints2_updated, Matches, outImg=None,
                                 matchColor=(0, 155, 0), singlePointColor=(0, 255, 255), matchesMask=good_matches,
                                 flags=0)

    # Displaying the image
    cv2.imwrite('Match.jpg', Matched)
    return matched_points1, matched_points2


if __name__ == "__main__":
    # Parameters for the local descriptor
    r_min = 5
    r_max = 20
    r_step = 1
    num_per_circle = 8
    matrix_size = (r_max - r_min) // r_step

    # Parameter for the descriptorMatching
    percentage_thresh = 0.3

    # Load and Detect the corners on the first image
    image1 = cv2.imread("im1.png")
    grayscale1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    if debug:
        filtered_corner_coords, descriptors = preProcessCorners(image1, grayscale1, r_min, r_max, r_step, num_per_circle, matrix_size)
        img1 = {"corners": filtered_corner_coords, "descriptor": descriptors}
        np.save('img1.npy', img1)
    else:
        img1 = np.load('img1.npy', allow_pickle=True).item()

    # Load and Detect the corners on the second image
    image2 = cv2.imread("im2.png")
    grayscale2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    if debug:
        filtered_corner_coords, descriptors = preProcessCorners(image2, grayscale2, r_min, r_max, r_step, num_per_circle, matrix_size)
        img2 = {"corners": filtered_corner_coords, "descriptor": descriptors}
        np.save('img2.npy', img2)
    else:
        img2 = np.load('img2.npy', allow_pickle=True).item()

    # Match the descriptors
    comb_image = cv2.imread("combined.png")
    # matchingPoints = descriptorMatching(img1, img2, percentage_thresh)
    matched_points1, matched_points2 = myDescriptorMatching(img1, img2, percentage_thresh, image1, image2)

    # for match in matchingPoints:
    #     cv2.line(comb_image, (int(img1["corners"][int(match[0])][0]), int(img1["corners"][int(match[0])][1])),
    #              (int(img2["corners"][int(match[1])][0]) + grayscale1.shape[1], int(img2["corners"][int(match[1])][1])),
    #              (0, 255, 0), 1)
    # for index, match in enumerate(matched_points1):
    #     x1, y1 = matched_points2[index]
    #     x2, y2 = matched_points2[index]

    cv2.imwrite("matchingPoints.jpg", comb_image)
