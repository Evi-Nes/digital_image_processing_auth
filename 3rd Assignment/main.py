import numpy as np
import cv2

debug = False

def myLocalDescriptor(img, p, rhom, rhoM, rhostep, N):
    """
    Computes the local descriptor for each pixel in the image
    :param img: the given image
    :param p: the given pixel
    :param rhom: the minimum radius
    :param rhoM: the maximum radius
    :param rhostep: the step of the radius
    :param N: the number of points in the circle
    :return: descriptor
    """

    if p[0] + rhoM > img.shape[0] or p[1] + rhoM > img.shape[1] or p[0] - rhoM < 0 or p[1] - rhoM < 0:
        d = np.zeros((1, 15))
        return d

    d = np.array([])
    for rho in range(rhom, rhoM, rhostep):
        x_rho = []
        for theta in range(0, 360, N):
            x = int(p[0] + rho * np.cos(theta))
            y = int(p[1] + rho * np.sin(theta))
            x_rho.append(img[x, y])
        d = np.append(d, np.mean(x_rho))

    return d

def myLocalDescriptorUpgrade(img, p, rhom, rhoM, rhostep, N):
    """
    Computes the local descriptor for each pixel in the image, based on our ideas
    :param img: the given grayscale image
    :param p: the given pixel
    :param rhom: the minimum radius
    :param rhoM: the maximum radius
    :param rhostep: the step of the radius
    :param N: the number of points in the circle
    :return: descriptor
    """
    d = np.array([])
    if p[0] + rhom > img.shape[0] or p[1] + rhom > img.shape[1] or p[0] - rhom < 0 or p[1] - rhom < 0:
        return d

    for rho in range(rhom, rhoM, rhostep):
        x_rho = []
        for theta in range(0, 360, N):
            if theta % (2 * N):
                x = int(p[0] + rho * np.cos(theta))
                y = int(p[1] + rho * np.sin(theta))
                x_rho.append(img[x, y] * 2)
            else:
                x = int(p[0] + rho * np.cos(theta))
                y = int(p[1] + rho * np.sin(theta))
                x_rho.append(img[x, y])

        d = np.append(d, np.mean(x_rho))

    return d

def myDetectHarrisFeatures(img, display_img):
    """
    Detects all the corners in the given image using the derivatives of x-axis and y-axis.
    :param img: the given grayscale image
    :param display_img: the given image udes for cv2.circle
    :return: the detected corners
    """
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    k = 0.04
    r_thresh = 0.4
    offset = 5
    height = img.shape[0]
    width = img.shape[1]
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
                cv2.circle(display_img, (x, y), 3, (0, 255, 0))

    cv2.imwrite("my_corners_img.jpg", display_img)
    return cornerList

def descriptorMatching(p1, p2, threshold):
    """
    Matches the descriptors of two images and returns the 30% of the matched points
    :param p1: the dictionary of first image
    :param p2: the dictionary of second image
    :param threshold: the percentage of the matched points we want to return
    :return: the matched points
    """
    matches = []
    used_indexes = np.array([])

    for i, point1 in enumerate(p1["corners"]):
        if all(value == 0 for value in p1["descriptor"][i]):
            continue

        min1 = np.inf
        index = 0

        for j, point2 in enumerate(p2["corners"]):
            if all(value == 0 for value in p2["descriptor"][j]):
                continue

            if j in used_indexes:
                continue

            dist_sum = np.sum(np.abs(p1["descriptor"][i] - p2["descriptor"][j]))

            if dist_sum < min1:
                min1 = dist_sum
                index = j

        used_indexes = np.append(used_indexes, index)
        matches.append([i, index])

    matches = np.array(matches)
    # matches = matches[matches[:, 1] != 0]
    # num_to_extract = int(matches.size * threshold)
    # matches = matches[:num_to_extract, :]

    return matches


if __name__ == "__main__":
    # Process the first image ########
    image1 = cv2.imread("imForest1.png")
    grayscale1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

    if debug:
        img1 = {"corners": myDetectHarrisFeatures(grayscale1, image1)}
        print(len(img1["corners"]))
        descriptor = np.zeros((len(img1["corners"]), 15))

        for i, point in enumerate(img1["corners"]):
            descriptor[i, :] = myLocalDescriptor(grayscale1, point, 5, 20, 1, 8)

        img1["descriptor"] = descriptor
        np.save('img1.npy', img1)
    else:
        img1 = np.load('img1.npy', allow_pickle=True).item()
        print(len(img1["corners"]))

    # Process the second image ########
    image2 = cv2.imread("imForest2.png")
    grayscale2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    if debug:
        img2 = {"corners": myDetectHarrisFeatures(grayscale2, image2)}
        print(len(img2["corners"]))
        descriptor = np.zeros((len(img2["corners"]), 15))

        for i, point in enumerate(img2["corners"]):
            descriptor[i, :] = myLocalDescriptor(grayscale2, point, 5, 20, 1, 8)

        img2["descriptor"] = descriptor
        np.save('img2.npy', img2)
    else:
        img2 = np.load('img2.npy', allow_pickle=True).item()
        print(len(img2["corners"]))

    # points = np.array([100, 100], [200, 200], [202, 202])
    # for point in points:
    #     descriptor = myLocalDescriptor(grayscale, point, 5, 20, 1, 8)
    #     descriptorUp = myLocalDescriptorUpgrade(grayscale, point, 5, 20, 1, 8)

    # Matching the descriptors ########
    percentage_thresh = 0.3
    matchingPoints = descriptorMatching(img1, img2, percentage_thresh)

    comb_image = cv2.imread("comb_forest.png")
    for match in matchingPoints:
        cv2.line(comb_image, (int(img1["corners"][int(match[0])][0]), int(img1["corners"][int(match[0])][1])),
                 (int(img2["corners"][int(match[1])][0]) + grayscale1.shape[0], int(img2["corners"][int(match[1])][1])), (0, 255, 0), 1)

    cv2.imwrite("matchingPoints.jpg", comb_image)
