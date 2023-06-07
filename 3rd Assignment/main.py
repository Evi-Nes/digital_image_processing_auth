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

def myDetectHarrisFeatures(img, display_img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    k = 0.04
    r_thresh = 0.3
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

    cv2.imwrite("my_corners.jpg", display_img)
    return cornerList


if __name__ == "__main__":
    image = cv2.imread("im1.png")
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # descriptor = myLocalDescriptor(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [202, 202], 5, 20, 1, 8)

    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [202, 202], 5, 20, 1, 8)

    # corners = myDetectHarrisFeatures(grayscale, image)
    corners = myDetectHarrisFeatures(grayscale, image)
    # print(corners)
    print(len(corners))

