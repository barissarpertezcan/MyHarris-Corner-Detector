import cv2 as cv
import numpy as np


def computeImageDerivative(img, ksize):
    """
    The function to calculate image derivatives with respect to x and y axes, their square and multiplication by using Sobel operator
  
    Parameters:
        img (2-D numpy array): The image whose image derivatives are going to be calculated
        ksize (int): Parameter for Sobel operator, i.e, blocksize on which the Sobel operator calculates the image derivative

    Returns:
        gxgy (2-D numpy array): Multiplication of image derivative with respect to x and y axes
        gx_square (2-D numpy array): Square of image derivative with respect to the x axis
        gy_square (2-D numpy array): Square of image derivative with respect to the y axis
    """

    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize)
    gxgy = gx * gy
    gx_square = np.power(gx, 2)
    gy_square = np.power(gy, 2)
    return gxgy, gx_square, gy_square

def getStructureTensor(coord, blocksize, gxgy, gx_square, gy_square):
    """
    The function to find Structure Tensor with respect to specified coordinate and blocksize
  
    Parameters:
        coord (tuple/list): The (x, y) coordinate of the point whose Structure Tensor is going to be found
        blocksize (int): Specifies the neighborhood size while computing Structure Tensor
        gxgy (2-D numpy array): Multiplication of image derivative with respect to x and y axes
        gx_square (2-D numpy array): Square of image derivative with respect to the x axis
        gy_square (2-D numpy array): Square of image derivative with respect to the y axis
  
    Returns:
        M (2-D numpy array): Structure Tensor of the given point
    """
    
    interval = int((blocksize - 1) / 2)

    i_x_square = np.sum(gx_square[coord[1] - interval: coord[1] + interval + 1, coord[0] - interval: coord[0] + interval + 1])
    i_y_square = np.sum(gy_square[coord[1] - interval: coord[1] + interval + 1, coord[0] - interval: coord[0] + interval + 1])
    i_xy = np.sum(gxgy[coord[1] - interval: coord[1] + interval + 1, coord[0] - interval: coord[0] + interval + 1])

    M = np.array([[i_x_square, i_xy], [i_xy, i_y_square]])
    return M

def computeHarrisScore(M, k):
    """
    The function to compute Harris Score for the point given its Structure Tensor
  
    Parameters:
        M (2-D numpy array): Structure Tensor of the given point
        k (float): Harris detector free parameter
        
    Returns:
        R (float): Harris Score of the given point
    """

    det_M = np.linalg.det(M)
    tr_M = np.trace(M)
    R = det_M - k * tr_M ** 2
    return R

def nonmaxsuppression(score_matrix, corner_coords, blocksize):
    """
    The function to apply Non-maximum Suppression on Harris Score of given corner coordinates over a neighborhood
  
    Parameters:
        score_matrix (2-D numpy array): Corner scores of points
        corner_coords (list): (x, y) coordinates of found corners
        blocksize (int): Specifies the neighborhood size while applying Non-maximum Suppression
        
    Returns:
        final_corner_coords (list): (x, y) coordinates of updated corners
    """

    final_corner_coords = []

    interval = int((blocksize - 1) / 2)

    for columns, rows in corner_coords:
        row, column = np.unravel_index(np.argmax(score_matrix[rows - interval: rows + interval + 1, columns - interval: columns + interval + 1]), (blocksize, blocksize))
        final_corner_coords.append((column + columns, row + rows))

    final_corner_coords = list(set(final_corner_coords))
    return final_corner_coords

def drawCorners(img, coords, showimg = False):
    """
    The function to draw detected corners on the image
  
    Parameters:
        img (3-D numpy array): The image over which corners are found
        coords (list): Stores (x, y) coordinates of corners
        showimg (boolean): Specifies whether or not img with drawn corners is going to be shown
    """

    for i, j in coords:
        cv.circle(img, (j, i), 1, [255, 0, 0], -1)

    if showimg:
        cv.imshow("corners", img)
        cv.waitKey(0)

def MyHarrisCorner(img, blocksize, ksize, k):
    """
    The function to implement Harris Corner Detector
  
    Parameters:
        img (2-D numpy array): The image whose image derivative is going to be calculated
        blocksize (int): Specifies the neighborhood size while applying Non-maximum Suppression
        ksize (int): Parameter for Sobel operator, i.e, blocksize on which the Sobel operator calculates the image derivative
        k (float): Harris detector free parameter
        
    Returns:
        score_matrix (2-D numpy array): Corner scores of points
        final_corner_coords (list): (x, y) coordinates of detected corners
    """

    # Applying Gaussian Blur to remove noises
    img = cv.GaussianBlur(img, (blocksize, blocksize), 0)
    gxgy, gx_square, gy_square = computeImageDerivative(img, ksize)
    score_matrix = np.zeros((img.shape[0], img.shape[1]))

    interval = int((blocksize - 1) / 2)
    for i in range(interval, img.shape[0] - interval):
        for j in range(interval, img.shape[1] - interval):
            M = getStructureTensor((j, i), blocksize, gxgy, gx_square, gy_square)
            R = computeHarrisScore(M, k)
            score_matrix[i, j] = R

    corner_coords = np.argwhere(score_matrix > 0.01 * score_matrix.max())
    final_corner_coords = nonmaxsuppression(score_matrix, corner_coords, blocksize)
    return score_matrix, final_corner_coords
    