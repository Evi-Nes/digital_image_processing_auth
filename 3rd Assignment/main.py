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

def isCorner(img, p, k, r_thresh, dx, dy):
    """
    Detects if the given pixel is a corner or not
    :param p: the given pixel
    :param k: the k parameter from the equation
    :param r_thresh: the threshold
    :param dx: the derivative in x-axis
    :param dy: the derivative in y-axis
    :return: true or false if the pixel is a corner or not
    """
    def w_exp(x, y):
        sigma = 1
        return np.exp(-(x**2 + y**2) / 2*sigma**2)

    # sigma = 1
    # w = np.exp(-(p[0]**2 + p[1]**2) / 2*sigma**2)
    # A = np.array([[dx[p[0], p[1]]**2, dx[p[0], p[1]]*dy[p[0], p[1]]], [dx[p[0], p[1]]*dy[p[0], p[1]], dy[p[0], p[1]]**2]])

    A = np.zeros((2, 2))
    A[0, 0] = dx[p[0], p[1]] * dx[p[0], p[1]]
    A[0, 1] = dx[p[0], p[1]] * dy[p[0], p[1]]
    A[1, 0] = dx[p[0], p[1]] * dy[p[0], p[1]]
    A[1, 1] = dy[p[0], p[1]] * dy[p[0], p[1]]
    # M = np.zeros((10, 10, 2, 2))

    # for u1 in range(-5, 6):
    #     for u2 in range(-5, 6):
    #         w = w_exp(u1, u2)
    #         M[u1, u2, :, :] = w * A

    sigma = 1.0
    M = cv2.GaussianBlur(A, (3, 3), sigma)

    R = np.linalg.det(M) - k * np.trace(M) ** 2

    return R > r_thresh

def myDetectHarrisFeatures(img, display_img):
    """
    For every pixel in the image, it calls the isCorner function and detects the corners.
    param img: the given grayscale image
    :return: the coordinates of the detected corners
    """
    detected_corners = []
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy
    # cv2.imshow('sobelx', dx)
    # cv2.waitKey(0)
    # cv2.imshow('sobely', dy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # laplacian = cv2.Laplacian(img,cv2.CV_64F)
    # cv2.imshow('laplacian',laplacian)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    k = 0.04
    r_thresh = 0.2

    for x, y in np.ndindex(img.shape):
        if isCorner(img, [x, y], k, r_thresh, dx2, dy2, dxy):
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), 2)
            detected_corners.append([x, y])

    cv2.imwrite("corners.jpg", display_img)

    return detected_corners

def my_harris(img, display_img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    k = 0.04
    r_thresh = 0.3
    height = img.shape[0]
    width = img.shape[1]
    matrix_R = np.zeros((height, width))

    #   Step 1 - Calculate the x e y image derivatives (dx e dy)
    dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=5)

    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    offset = 3
    #   Step 3 - Calcular a soma dos produtos das derivadas para cada pixel (Sx2, Sy2 e Sxy)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(dx2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det - k * (tr ** 2)
            matrix_R[y - offset, x - offset] = R

    cornerList = []
    #   Step 6 - Apply a threshold
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrix_R[y, x]
            if value > r_thresh:
                cornerList.append([x, y, value])
                cv2.circle(display_img, (x, y), 3, (0, 255, 0))

    cv2.imwrite("my_corners.jpg", display_img)
    return cornerList


if __name__ == "__main__":
    image = cv2.imread("imForest1.png")
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # descriptor = myLocalDescriptor(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [202, 202], 5, 20, 1, 8)

    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptorUp = myLocalDescriptorUpgrade(grayscale, [202, 202], 5, 20, 1, 8)

    # corners = myDetectHarrisFeatures(grayscale, image)
    corners = my_harris(grayscale, image)
    # print(corners)
    print(len(corners))

