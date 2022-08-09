import cv2 as cv
import numpy as np


def image_derivatives(img, kernel_type = "scharr"):
    # Apply Gaussian blur to image to remove noise
    img = cv.GaussianBlur(img, (5,5), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ksize = -1 if kernel_type == "scharr" else 3
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize).ravel()
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize).ravel()
    # print(gx.shape)
    # print(gy.shape)
    gradient_vectors = np.stack((gx, gy), axis=1)
    return gx, gy, gradient_vectors

def manual_test(img, coord, kernel_type):
    img = cv.GaussianBlur(img, (5,5), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    north, south, east, west = img[coord[1] - 1, coord[0]], img[coord[1] + 1, coord[0]], img[coord[1], coord[0] + 1], img[coord[1], coord[0] - 1]
    north_west, north_east, south_east, south_west = img[coord[1] - 1, coord[0] - 1], img[coord[1] - 1, coord[0] + 1], img[coord[1] + 1, coord[0] + 1], img[coord[1] + 1, coord[0] - 1]
    
    if kernel_type == "sobel":
        diagonal_weight = 1
        side_weight = 2
    else:
        diagonal_weight = 3
        side_weight = 10

    gx = (west - east) * side_weight + (north_west - north_east) * diagonal_weight + (south_west - south_east) * diagonal_weight
    gy = (north - south) * side_weight + (north_west - south_west) * diagonal_weight + (north_east - south_east) * diagonal_weight
    
    return gx, gy


img = cv.imread("imgs/edgy_img2.png")
img = cv.resize(img, (15, 15), cv.INTER_AREA)
# cv.imshow("img", img)
# cv.waitKey(0)
gx, gy, gradient_vectors = image_derivatives(img)
# print(gx.shape)
# print(gy.shape)
print(gradient_vectors)
print(gradient_vectors.shape)


"""
# manual image derivative test
coord = [1, 1]
manual_gx, manual_gy = manual_test(img, coord, "scharr")

print("----manual-----")
print(manual_gx)
print(manual_gy)

print("----computed----")
print(gx[1, 1])
print(gy[1, 1])
"""

"""
# direction test

img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.uint8)
coord = [1, 1]
north, south, east, west = img[coord[1] - 1, coord[0]], img[coord[1] + 1, coord[0]], img[coord[1], coord[0] + 1], img[coord[1], coord[0] - 1]
north_west, north_east, south_east, south_west = img[coord[1] - 1, coord[0] - 1], img[coord[1] - 1, coord[0] + 1], img[coord[1] + 1, coord[0] + 1], img[coord[1] + 1, coord[0] - 1]

print(north, south, east, west)
print(north_west, north_east, south_east, south_west)
"""
