import numpy as np
import cv2
def myLocalDescriptor (I, p, rhom, rhoM, rhostep, N):
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


if __name__ == "__main__":
    image = cv2.imread("image222.png")
    display_image = np.copy(image)

    grayscale = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
    myLocalDescriptor(grayscale, 1, 1, 1, 1, 1)