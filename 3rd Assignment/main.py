import numpy as np
import cv2
from scipy.ndimage import convolve
def myLocalDescriptor(I, p, rhom, rhoM, rhostep, N):
    """
    Computes the local descriptor for each pixel in the image
    :param I: the given image
    :param p: the given pixel
    :param rhom: the minimum radius
    :param rhoM: the maximum radius
    :param rhostep: the step of the radius
    :param N: the number of points in the circle
    :return: descriptor
    """
    d = np.array([])
    if p[0] + rhom > I.shape[0] or p[1] + rhom > I.shape[1] or p[0] - rhom < 0 or p[1] - rhom < 0:
        return d

    for rho in range(rhom, rhoM, rhostep):
        x_rho = []
        for theta in range(0, 360, N):
            x = int(p[0] + rho * np.cos(theta))
            y = int(p[1] + rho * np.sin(theta))
            x_rho.append(I[x, y])
        d = np.append(d, np.mean(x_rho))
    # print(d)
    # print("\n")
    return d

def myLocalDescriptorUpgrade(I, p, rhom, rhoM, rhostep, N):
    """
    Computes the local descriptor for each pixel in the image, based on our ideas
    :param I: the given image
    :param p: the given pixel
    :param rhom: the minimum radius
    :param rhoM: the maximum radius
    :param rhostep: the step of the radius
    :param N: the number of points in the circle
    :return: descriptor
    """
    log_size = 5  # Size of the LoG matrix
    sigma = 1.0
    # Generate the LoG matrix
    gaussian = cv2.getGaussianKernel(log_size, sigma)
    log_matrix = cv2.sepFilter2D(gaussian, -1,  np.transpose(gaussian), np.transpose(gaussian))

    # Convolve the image with the LoG matrix
    convolved_image = convolve(I, log_matrix)
    cv2.imwrite("convolved_image2.jpg", convolved_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return convolved_image


if __name__ == "__main__":
    image = cv2.imread("im2.png")
    display_image = np.copy(image)
    grayscale = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)

    descriptor = myLocalDescriptor(grayscale, [100, 100], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [200, 200], 5, 20, 1, 8)
    # descriptor = myLocalDescriptor(grayscale, [202, 202], 5, 20, 1, 8)

    myLocalDescriptorUpgrade(grayscale, [100, 100], 5, 20, 1, 8)
