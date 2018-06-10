# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def getbg(img, msk):
    return (img*msk).astype('uint8')


oldcammtx = np.float32([[2.32729520e+03, 0.00000000e+00, 1.92959380e+03],
                        [0.00000000e+00, 2.32850245e+03, 1.05881793e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
olddistcoeff = np.float32([-0.14374176,  0.13863309, -0.00090426,  0.00109337, -0.04210762])

camera_intrinsic_matrix = np.float32([[2.21891528e+03,  0.00000000e+00,   1.93571064e+03],
                                      [0.00000000e+00,  2.26785645e+03,   1.05642772e+03],
                                      [0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])

dist_coeffs = np.zeros((4,1), np.float32)


mask = np.ones((2160,3840,3))
mask[:,1450:2650] = 0
mask[:700] = 0
mask[:850,2000:] = 0
mask[-500:-70, :800] = 0
mask[1200:1309, 1100:1218] = 0
mask[1186:1470, 2650:3230] = 0
# img1 = (img1*mask).astype('uint8')
# img2 = (img2*mask).astype('uint8')
mask6 = np.ones((2160, 3840, 3))
mask6[:,1171:2304] = 0
mask6[:700,2304:] = 0
mask6[:580,:1171] = 0
mask6[1536:1620,:81] = 0
mask6[1185:1521,2480:3260] = 0
mask6[1304:1404,620:672] = 0

sift_params = dict(nfeatures = 0,nOctaveLayers = 3,contrastThreshold = 0.05,edgeThreshold = 8,sigma = 1.26)
sift = cv2.xfeatures2d.SIFT_create(**sift_params)

# bg_file1 = '/Users/yujieli/Documents/CFS_Video_Analysis-master/NFramesRaw5/frame270.png'
# img1 = cv2.imread(bg_file1)
# img1 = getbg(img1,mask)
# img1 = cv2.undistort(img1, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# # cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/undisttest270.png', img1)
# kp1, des1 = sift.detectAndCompute(img1, None)

# bg_file2 = '/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ6/frame1.png'
# img2 = cv2.imread(bg_file2)
# # img2_Cut = getbg(img2,mask)
# img2 = cv2.undistort(img2, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# # cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/undisttest2.png', img2)
# kp2, des2 = sift.detectAndCompute(img2, None)

bg_file1 = '/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ6/frame1.png'
img1 = cv2.imread(bg_file1)
img1_cut = getbg(img1,mask6)
img1 = cv2.undistort(img1, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/undisttest1.png', img1)
kp1, des1 = sift.detectAndCompute(img1_cut, None)

bg_file2 = '/Users/yujieli/Documents/CFS_Video_Analysis-master/NFramesRaw5/frame270.png'
img2 = cv2.imread(bg_file2)
img2_cut = getbg(img2,mask)
img2_cut = cv2.undistort(img2_cut, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/undisttest2.png', img2)
kp2, des2 = sift.detectAndCompute(img2_cut, None)

eq6001 = np.float32([[1287,345], [2295,331], [2237,1940], [1418,1935],
    [1981,534], [2132,533], [2131,677], [1981,677],
    [1983,820], [2130,820], [2128,956], [1982,956], 
    [1983,1090],[2125,1090],[2123,1217],[1983,1218],
    [1983,1342],[2120,1342],[2119,1460],[1983,1460],
    [1983,1576],[2116,1576],[2114,1686],[1983,1686],
    [1983,1794],[2111,1794],[2110,1893],[1983,1894],
    [1454,541], [1624,539],
    [1470,825], [1635,823],
    [1485,1092],[1647,1092],
    [1503,1343],[1658,1342],
    [1517,1574],[1670,1575],
    [1534,1791],[1681,1791]
    ]).reshape(-1,1,2)
# Default params
# int     nfeatures = 0,
# int     nOctaveLayers = 3,
# double  contrastThreshold = 0.04,
# double  edgeThreshold = 10,
# double  sigma = 1.6
# sift_params = dict(nfeatures = 0,nOctaveLayers = 3,contrastThreshold = 0.07,edgeThreshold = 10,sigma = 1)
# sift = cv2.xfeatures2d.SIFT_create(**sift_params)
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

MIN_MATCH_COUNT = 10

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    E ,mask0 = cv2.findEssentialMat(src_pts, dst_pts, camera_intrinsic_matrix)
    M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    F, mask2 = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 1., 0.999)

    print 'Good pairs:', np.nonzero(mask0)[0].shape[0]

    matchesMask = mask0.ravel().tolist()

    h,w,cc = img1.shape
    pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(
                    singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                flags = 2)

img3 = cv2.drawMatches(img1_cut,kp1,img2_cut,kp2,good,None,**draw_params)

plt.imshow(img3, cmap='gray'),plt.show()


