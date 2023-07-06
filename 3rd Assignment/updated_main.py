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
        for index2, corner2 in enumerate(corners2):
            descriptor2 = np.array(descriptors2[index2])

            if descriptor1.any() == 1e20 or descriptor2.any() == 1e20:
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
    corners2, descriptors2 = p2["corners"], p2["descriptor"]

    # calculateDistances(corners1, corners2, descriptors1, descriptors2)

    distances = np.load("distances.npy")
    matched_points = []
    matched_coords = []

    for index, corner in enumerate(corners1):
        one_corner_distances = distances[index]
        sorted_distances = np.argsort(one_corner_distances)
        filtered_distances = sorted_distances[:int(thresh * len(sorted_distances))]
        for filtered in filtered_distances:
            matched_points.append((index, filtered))
            matched_coords.append((corners1[index], corners2[filtered]))

    return matched_coords

def calculate_rho_theta(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1

    rho = np.sqrt(delta_x ** 2 + delta_y ** 2)
    theta = np.arctan2(delta_y, delta_x)
    theta = np.degrees(theta)

    return rho, theta

def getCoordinates(matching_points, img1, img2):
    matched1 = []
    matched2 = []
    for match in matching_points:
        x1, y1 = match[0]
        x2, y2 = match[1]
        matched1.append((x1, y1))
        matched2.append((x2, y2))

    return matched1, matched2

def getTransformedPoints(points, d, theta):
    """
    Gets the matched points and calculates the transformed points
    """
    transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), d],
                                      [np.sin(theta), np.cos(theta), 0],
                                      [0, 0, 1]])
    points = np.array(points)
    homogeneous_points = np.column_stack((points[:, 0], np.ones(len(points))))
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]

    return transformed_points

def myRansac(matched_points, img1, img2):
    """
    Gets the matched points and compares random pairs to find the optimal transformation matrix
    :param matched_points:
    :return:
    """
    suffled_points = np.copy(matched_points)
    random.shuffle(suffled_points)
    suffled_points = suffled_points[:100]
    rho_theta = []
    points = img1['corners'] + img2['corners']

    for index, match in enumerate(suffled_points):
        im1_x1, img1_y1 = suffled_points[index][0]
        im2_x1, im2_y1 = suffled_points[index][1]
        im1_x2, im1_y2 = suffled_points[index+1][0]
        im2_x2, im2_y2 = suffled_points[index+1][1]
        rho1, theta1 = calculate_rho_theta(im1_x1, img1_y1, im2_x1, im2_y1)
        rho2, theta2 = calculate_rho_theta(im1_x2, im1_y2, im2_x2, im2_y2)
        rho = (rho1 + rho2) // 2
        theta = (theta1 + theta2) // 2
        rho_theta.append((rho, theta))

        getTransformedPoints(points, rho, theta)


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
    # comb_image = cv2.imread("combined.png")
    matchingPoints = descriptorMatching(img1, img2, percentage_thresh)
    # matched_points1, matched_points2 = myDescriptorMatching(img1, img2, percentage_thresh, image1, image2)
    # matched_points1, matched_points2 = getCoordinates(matchingPoints, img1, img2)
    myRansac(matchingPoints, img1, img2)

    rho_theta = []
    for index, match in enumerate(matched_points1):
        x1, y1 = matched_points1[index]
        x2, y2 = matched_points2[index]
        rho, theta = calculate_rho_theta(x1, y1, x2, y2)
        rho_theta.append((rho, theta))

    rho_theta = np.array(rho_theta)
    mean_values = np.mean(rho_theta, axis=0)
    print('mean', mean_values)

    # Apply K-means clustering
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(rho_theta)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    similar_values = []

    for i in range(num_clusters):
        cluster_points = rho_theta[labels == i]
        average_rho = np.mean(cluster_points[:, 0])
        average_theta = np.mean(cluster_points[:, 1])
        similar_values.append((average_rho, average_theta))

    print(similar_values)
