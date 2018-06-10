import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
cv2.namedWindow('check', cv2.WINDOW_NORMAL)
cv2.resizeWindow('check', 1000, 1000)
images = glob.glob('/Users/yujieli/Documents/CFS_Video_Analysis-master/YujieLi_VideoAnalysis/Calib/*.png')
count = 0
for fname in images:

    img = cv2.imread(fname)
    print img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,8), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6,8), corners2,ret)
        count += 1
    print count


# cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print mtx
print dist
print count

img = cv2.imread('/Users/yujieli/Documents/CFS_Video_Analysis-master/YujieLi_VideoAnalysis/Calib/frame3499.png')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
print newcameramtx
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
print roi

# undistort
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/calib2222.png',dst)
