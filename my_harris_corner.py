import cv2 as cv
from matplotlib.pyplot import draw
import numpy as np

# docstring yazılacak

def computeImageDerivative(img, ksize):
    """
    The function to calculate image derivatives by using Sobel operator
  
    Parameters:
        img (2-D numpy array): The image whose image derivative is calculated
        ksize (int): Parameter for Sobel operator, i.e, blocksize on which the Sobel operator calculate the image derivative

    Returns:
        gx (2-D numpy array): Image derivatives with respect to x axis
        gy (2-D numpy array): Image derivatives with respect to y axis
    """

    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize)
    return gx, gy

def getStructureTensor(coord, blocksize, gx, gy):
    interval = int((blocksize - 1) / 2)

    roi_x = gx[coord[1] - interval: coord[1] + interval + 1, coord[0] - interval: coord[0] + interval + 1]
    roi_y = gy[coord[1] - interval: coord[1] + interval + 1, coord[0] - interval: coord[0] + interval + 1]
    i_x_square = np.sum(np.power(roi_x, 2))
    i_y_square = np.sum(np.power(roi_y, 2))
    i_xy = np.sum(roi_x * roi_y)

    M = np.array([[i_x_square, i_xy], [i_xy, i_y_square]])
    return M

def computeHarrisScore(M, k):
    det_M = np.linalg.det(M)
    tr_M = np.trace(M)
    R = det_M - k * tr_M ** 2
    return R

def nonmaxsuppression(score_matrix, corner_coords, blocksize):
    final_corner_coords = []

    interval = int((blocksize - 1) / 2)

    for columns, rows in corner_coords:
        row, column = np.unravel_index(np.argmax(score_matrix[rows - interval: rows + interval + 1, columns - interval: columns + interval + 1]), (blocksize, blocksize))
        final_corner_coords.append((column + columns, row + rows))

    final_corner_coords = list(set(final_corner_coords))
    return final_corner_coords

def drawCorners(img, coords, showimg = False):
    for i, j in coords:
        cv.circle(img, (j, i), 1, [255, 0, 0], -1)

    if showimg:
        cv.imshow("corners", img)
        cv.waitKey(0)

def MyHarrisCorner(img, blocksize, ksize, k):
    # Applying Gaussian Blur to remove noises
    img = cv.GaussianBlur(img, (blocksize, blocksize), 0)
    gx, gy = computeImageDerivative(img, ksize)
    score_matrix = np.zeros((img.shape[0], img.shape[1]))

    interval = int((blocksize - 1) / 2)
    for i in range(interval, img.shape[0] - interval):
        for j in range(interval, img.shape[1] - interval):
            M = getStructureTensor((j, i), blocksize, gx, gy)
            R = computeHarrisScore(M, k)
            score_matrix[i, j] = R

    corner_coords = np.argwhere(score_matrix > 0.01 * score_matrix.max())
    final_corner_coords = nonmaxsuppression(score_matrix, corner_coords, blocksize)
    return score_matrix, final_corner_coords
    
