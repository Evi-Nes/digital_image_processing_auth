import numpy as np
import cv2
from scipy.ndimage import convolve
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
    d = np.array([])
    if p[0] + rhom > img.shape[0] or p[1] + rhom > img.shape[1] or p[0] - rhom < 0 or p[1] - rhom < 0:
        return d

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
            if theta % (2*N):
                x = int(p[0] + rho * np.cos(theta))
                y = int(p[1] + rho * np.sin(theta))
                x_rho.append(img[x, y]*2)
            else:
                x = int(p[0] + rho * np.cos(theta))
                y = int(p[1] + rho * np.sin(theta))
                x_rho.append(img[x, y])

        d = np.append(d, np.mean(x_rho))

    return d

def isCorner(p, k, r_thresh, dx, dy):
    """
    Detects if the given pixel is a corner or not
    :param p: the given pixel
    :param k: the k parameter from the equation
    :param r_thresh: the threshold
    :param dx: the derivative in x-axis
    :param dy: the derivative in y-axis
    :return: true or false if the pixel is a corner or not
    """
    sigma = 1
    w = np.exp(-(p[0]**2 + p[1]**2) / 2*sigma**2)
    A = np.array([[dx[p[0], p[1]]**2, dx[p[0], p[1]]*dy[p[0], p[1]]], [dx[p[0], p[1]]*dy[p[0], p[1]], dy[p[0], p[1]]**2]])

    M = w * A
    R = np.linalg.det(M) - k * np.trace(M) ** 2

    if R > r_thresh:
        return True
    else:
        return False

def myDetectHarrisFeatures(img, display_img):
    """
    For every pixel in the image, it calls the isCorner function and detects the corners.
    :param img: the given grayscale image
    :return: the coordinates of the detected corners
    """
    detected_corners = []
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    for x, y in np.ndindex(img.shape):
        if isCorner([x, y], 4, 0.2, dx, dy):
            cv2.circle(display_img, (y, x), 5, (0, 0, 255), 2)
            detected_corners.append([x, y])

    cv2.imwrite("corners.jpg", display_img)

    return detected_corners


if __name__ == "__main__":
    image = cv2.imread("im2.png")
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # descriptor = myLocalDescriptor(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [202, 202], 5, 20, 1, 8)

    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [202, 202], 5, 20, 1, 8)

    corners = myDetectHarrisFeatures(grayscale, image)
    print(corners)
    print(len(corners))

