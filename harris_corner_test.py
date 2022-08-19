from my_harris_corner import *
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    img = cv.imread("imgs/book1.png")
    img = cv.resize(img, (500, 500), cv.INTER_AREA)
    colored_img = img.copy()
    colored_img2 = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    score_matrix, corner_coords = MyHarrisCorner(gray, 3, 3, 0.04)

    colored_img[score_matrix > 0.01 * score_matrix.max()]= [255, 0, 0]
    # drawCorners(colored_img, corner_coords)

    # opencv harris corner
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 3, 3, 0.04)
    # dst = cv.dilate(dst,None)
    colored_img2[dst>0.01*dst.max()]=[255, 0, 0]

    org = (20, gray.shape[0] - 20)
    colored_img = cv.putText(colored_img, 'My Harris', org, cv.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv.LINE_AA)

    colored_img2 = cv.putText(colored_img2, 'Original Harris', org, cv.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv.LINE_AA)

    comparison = np.hstack((colored_img2, colored_img))
    # cv.imwrite("comparison.png", comparison)
    cv.imshow("comparison", comparison)
    cv.waitKey(0)