import numpy as np
import cv2
from sklearn.cluster import KMeans
import random

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
    r_thresh = 0.25
    offset = 4
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
        descriptor1 = np.array(descriptors1[index1])
        if np.any(descriptor1 > 1000000):
            distances[index1, :] = 1e20
            continue
        for index2, corner2 in enumerate(corners2):
            descriptor2 = np.array(descriptors2[index2])
            if np.any(descriptor2 > 1000000):
                distances[index1, index2] = 1e20
            else:
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
    # corners2, descriptors2 = p2["corners"], p2["descriptor"]
    # calculateDistances(corners1, corners2, descriptors1, descriptors2)

    distances = np.load("distances.npy")
    matched_points = []

    for index, corner in enumerate(corners1):
        corner_distances = distances[index]
        min_value = min(corner_distances)
        min_index = np.argmin(corner_distances)
        matched_points.append((index, min_index, min_value))

    min_indices = sorted(matched_points, key=lambda x: x[2])
    min_indices = min_indices[:int(thresh * len(matched_points))]
    return min_indices
def calculate_rho_theta(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    theta = np.arctan2(delta_y, delta_x)
    theta = np.degrees(theta)

    return theta

def getTransformedPoints(matched_points, points1, d, theta):
    """
    Gets the matched points and calculates the transformed points
    """
    transformed_points = []
    cos_sin_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])

    for index, match in enumerate(matched_points):
        x1, y1 = points1[match[0]]
        point1 = np.array([x1, y1])

        transformed_point = np.dot(cos_sin_matrix.T, point1)
        transformed_point = transformed_point + d
        transformed_points.append(transformed_point)

    return transformed_points

def myRansac(matched_points, img1, img2, r_thresh):
    """
    Gets the matched points and compares random pairs to find the optimal transformation matrix
    :param matched_points:
    :return:
    """
    inliers = []
    outliers = []
    points1 = img1['corners']
    points2 = img2['corners']
    best_score = 0

    for index, match in enumerate(matched_points):
        next_index = index + 1
        if index == len(matched_points) - 1:
            next_index = 20

        im1_x1, im1_y1 = points1[matched_points[index][0]]
        im2_x1, im2_y1 = points2[matched_points[index][1]]
        im1_x2, im1_y2 = points1[matched_points[next_index][0]]
        im2_x2, im2_y2 = points2[matched_points[next_index][1]]

        theta1 = calculate_rho_theta(im1_x1, im1_y1, im1_x2, im1_y2)
        theta2 = calculate_rho_theta(im2_x1, im2_y1, im2_x2, im2_y2)
        theta = theta2 - theta1
        d1 = [(im1_x1-im2_x1), (im1_y1-im2_y1)]
        d2 = [(im1_x2-im2_x2), (im1_y2-im2_y2)]
        d = (np.array(d2) + np.array(d1)) // 2

        transformed_points = getTransformedPoints(matched_points, points1, d, theta)
        for index, point in enumerate(transformed_points):
            x_1, y_1 = points1[matched_points[index][0]]
            x_new, y_new = point
            distance = np.linalg.norm(np.array([x_1, y_1]) - np.array([x_new, y_new]))

            if distance < r_thresh:
                inliers.append(matched_points[index])
            else:
                outliers.append(matched_points[index])

        score = len(inliers) / len(matched_points)

        if score > best_score:
            best_score = score
            best_d = d
            best_theta = theta

    return best_d, best_theta, inliers, outliers


if __name__ == "__main__":
    # Parameters for the local descriptor
    r_min = 5
    r_max = 20
    r_step = 1
    num_per_circle = 8
    matrix_size = (r_max - r_min) // r_step

    # Parameter for the descriptorMatching
    percentage_thresh = 0.2

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
    r = 20
    matchingPoints = descriptorMatching(img1, img2, percentage_thresh)
    final_d, final_theta, final_inliers, final_outliers = myRansac(matchingPoints, img1, img2, r)
    print("Final D: ", final_d)
    print("Final Theta: ", final_theta)
