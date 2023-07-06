import numpy as np
import cv2

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
            # cv2.circle(img, (x, y), 1, (255, 0, 0))

        d[0, index] = np.mean(x_rho)
        index += 1
    # cv2.imwrite('myDescriptor.jpeg', img)
    return d

def myLocalDescriptorUpgrade(img, p, r_min, r_max, r_step, num_points):
    """
    Computes the local descriptor for each pixel in the image, based on our ideas
    :param img: the given grayscale image
    :param p: the given pixel
    :param r_min: the minimum radius
    :param r_max: the maximum radius
    :param r_step: the step of the radius
    :param num_points: the number of points in the circle
    :return: descriptor
    """
    size = (r_max - r_min) // r_step
    d = np.full((1, size), 1e20)

    if p[0] + r_max > img.shape[1] or p[1] + r_max > img.shape[0] or p[0] - r_max < 0 or p[1] - r_max < 0:
        return d

    index = 0
    for rho in range(r_min, r_max, r_step):
        x_rho = []
        for theta in range(0, 360, 360 // num_points):
            if (theta % 20) == 0:
                x = int(p[0] + rho * np.cos(theta))
                y = int(p[1] + rho * np.sin(theta))
                x_rho.append(img[x, y] * 2)
            else:
                x = int(p[0] + rho * np.cos(theta))
                y = int(p[1] + rho * np.sin(theta))
                x_rho.append(img[x, y])

            # cv2.circle(img, (x, y), 1, (255, 0, 0))
        d[0, index] = np.mean(x_rho)
        index += 1

    # cv2.imwrite('myDescriptorUpgrade.jpeg', img)
    return d


if __name__ == "__main__":
    # Parameters for the local descriptor
    r_min = 5
    r_max = 20
    r_step = 1
    num_per_circle = 8
    matrix_size = (r_max - r_min) // r_step

    # Load the image
    image = cv2.imread("im1.png")
    copyImg = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    points = np.array(([100, 100], [200, 200], [202, 202]))
    for point in points:
        descriptor = myLocalDescriptor(grayscale, point, r_min, r_max, r_step, num_per_circle)
        descriptorUp = myLocalDescriptorUpgrade(grayscale, point, r_min, r_max, r_step, num_per_circle)
        print('end')
