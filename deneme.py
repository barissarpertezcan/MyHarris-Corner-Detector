import cv2 as cv
import numpy as np

"""ar = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# print(ar)
img = cv.imread("book1.png")
# cv.imshow("original book", img)

img2 = cv.filter2D(img, -1, kernel=ar)
# cv.imshow("modified book", img)

final = np.hstack((img, img2))
cv.imshow("final", final)

cv.waitKey(0)
"""

"""filename = 'imgs/book1.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,3,3,0.04)
#result is dilated for marking the corners, not important
# dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[255,0,0]
print(np.sum(dst>0.01*dst.max()))

cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
"""

if __name__ == "__main__":
    print("hehe")