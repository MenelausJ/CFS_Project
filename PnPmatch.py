import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def getbg(img, msk):
    return (img*msk).astype('uint8')


def triangulate(P1, P2, matchp1, matchp2):

    points = np.zeros([4,matchp1.shape[0]])
    i = 0
    for mp1, mp2 in zip(matchp1, matchp2):
        A = np.vstack((mp1.flatten()[0]*P1[2]-P1[0],
                       mp1.flatten()[1]*P1[2]-P1[1],
                       mp2.flatten()[0]*P2[2]-P2[0],
                       mp2.flatten()[1]*P2[2]-P2[1]))
        u, s, v = np.linalg.svd(A)
        points[:,i] = v[-1]
        i += 1
#         Roughly same using eig
#         w, v = np.linalg.eig(A)
#         points.append(v[:3,-1]/v[3,-1])
    return points


def to_homog(points):
    return np.vstack((points,np.ones(points.shape[1])))


def from_homog(points_homog):
    return points_homog[:-1]/points_homog[-1]


# def undistort(pts, distcoeffs, oldcammat, newcammat, R):
#     ctx, cty = oldcammat[2, :2].flatten()
#     ctxp, ctyp = newcammat[2, :2].flatten()
#     k1, k2, k3, p1, p2 = distcoeffs.flatten()
#     newpts = np.zeros_like(pts)
#     for i in range(pts.shape[0]):
#         u, v = pts[i, ...].flatten()
#         xpp = (u - ctx) / oldcammat[0, 0]
#         ypp = (v - cty) / oldcammat[1, 1]
#
#         r = np.sqrt(xpp**2 + ypp**2)
#         xp = xpp * (1 + k1*r**2 + k2*r**4 + k3*r**6) + 2*p1*xpp*ypp + p2*(r**2 + 2*xpp**2)
#         yp = ypp * (1 + k1*r**2 + k2*r**4 + k3*r**6) + 2*p2*xpp*ypp + p1*(r**2 + 2*ypp**2)
#
#         xrot = np.dot(np.linalg.inv(R), np.array([xp, yp, 1]))
#         xp, yp = from_homog(xrot)
#
#         newpts[i, ..., 0] = xp * newcammat[0, 0] + ctxp
#         newpts[i, ..., 1] = yp * newcammat[1, 1] + ctyp
#
#
#     return newpts


# Initialize approximate camera intrinsic matrix

oldcammtx = np.float32([[2.32729520e+03, 0.00000000e+00, 1.92959380e+03],
                        [0.00000000e+00, 2.32850245e+03, 1.05881793e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
olddistcoeff = np.float32([-0.14374176,  0.13863309, -0.00090426,  0.00109337, -0.04210762])

camera_intrinsic_matrix = np.float32([[2.21891528e+03,  0.00000000e+00,   1.93571064e+03],
                                      [0.00000000e+00,  2.26785645e+03,   1.05642772e+03],
                                      [0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])

dist_coeffs = np.zeros((4,1), np.float32)
roi = (4, 24, 3832, 2109)

mask = np.ones((2160, 3840))
mask[:,1450:2650] = 0
mask[:720] = 0
mask[:850,2000:] = 0
mask[-500:-70, :800] = 0
mask[1200:1309, 1100:1218] = 0
mask[1186:1470, 2650:3230] = 0


mask7 = np.ones((2160, 3840))
mask7[:,1503:2689] = 0
mask7[:827] = 0
mask7[1245:1464,1284:1627] = 0
mask7[1215:1455,2656:3519] = 0
# h = 19.579m
# w = 7.56m
# l = 10.603m

# Top wall: 1195


bld = np.float32([[0, 0, 0], [10520, 0, 0],[10520, 19579, 0], [0, 19579, 0]])
building = np.zeros((40, 3), dtype='float32')
building[:4] = bld
for i in range(6):
    building[4*i+4:4*i+8] = np.float32([[7396,2100+i*3050,0],[8870,2100+i*3050,0],
        [8870,i*3050+3620,0],[7396,i*3050+3620,0]])
for i in range(6):
    building[28+2*i:28+2*i+2] = np.float32([[1650,2100+i*3050,0],[3583,2100+i*3050,0]])


building = (building+np.float32([0,0,30000])).reshape(-1,1,3)
# with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/building.txt", "w") as f1:
#     for i in range(building.shape[0]):
#         for j in range(3):
#             f1.write(str(building[i,:,j].squeeze()) + '\t')
#         f1.write("\n")

# EQ6001
# bld_img1 = np.float32([[1287,345], [2295,331], [2237,1940], [1418,1935],
#     [1981,534], [2132,533], [2131,677], [1981,677],
#     [1983,820], [2130,820], [2128,956], [1982,956], 
#     [1983,1090],[2125,1090],[2123,1217],[1983,1218],
#     [1983,1342],[2120,1342],[2119,1460],[1983,1460],
#     [1983,1576],[2116,1576],[2114,1686],[1983,1686],
#     [1983,1794],[2111,1794],[2110,1893],[1983,1894],
#     [1454,541], [1624,539],
#     [1470,825], [1635,823],
#     [1485,1092],[1647,1092],
#     [1503,1343],[1658,1342],
#     [1517,1574],[1670,1575],
#     [1534,1791],[1681,1791]
#     ]).reshape(-1,1,2)

# EQ5270
bld_img1 = np.float32([[1755,517], [2519,513], [2450,1774], [1791,1770],
    [2280,664], [2394,664], [2391,771], [2277,771],
    [2275,880], [2387,880], [2384,984], [2272,985],
    [2270,1088],[2379,1088],[2375,1187],[2266,1187],
    [2263,1285],[2370,1286],[2366,1379],[2260,1379],
    [2257,1473],[2361,1474],[2358,1563],[2253,1562],
    [2251,1652],[2352,1652],[2348,1735],[2248,1733],
    [1878,666], [2008,665],
    [1880,881], [2008,881],
    [1883,1087],[2009,1087],
    [1887,1283],[2009,1284],
    [1889,1470],[2009,1471],
    [1892,1648],[2009,1648]
    ]).reshape(-1,1,2)

# EQ71649
bld_img2 = np.float32([[1594,335], [2498,306], [2464,1847], [1673,1846],
    [2218,493], [2354,491], [2355,620], [2219,624],
    [2220,754], [2354,752], [2354,880], [2220,883],
    [2221,1007],[2352,1005],[2352,1127],[2221,1129],
    [2221,1249],[2349,1247],[2348,1363],[2221,1364],
    [2221,1480],[2346,1478],[2344,1588],[2220,1589],
    [2219,1699],[2341,1697],[2339,1800],[2218,1800],
    [1743,508], [1896,503],
    [1753,765], [1903,762],
    [1763,1014],[1911,1012],
    [1771,1254],[1918,1252],
    [1781,1483],[1924,1481],
    [1787,1699],[1930,1698]
    ]).reshape(-1,1,2)

undist1 = cv2.undistortPoints(bld_img1, oldcammtx, olddistcoeff, np.eye(3), camera_intrinsic_matrix)
undist2 = cv2.undistortPoints(bld_img2, oldcammtx, olddistcoeff, np.eye(3), camera_intrinsic_matrix)

retvl1, rvec1, tvec1 = cv2.solvePnP(building,
                                    undist1,
                                    camera_intrinsic_matrix,
                                    dist_coeffs)
Rmat1, jacobian1 = cv2.Rodrigues(rvec1)

retvl2, rvec2, tvec2 = cv2.solvePnP(building,
                                    undist2,
                                    camera_intrinsic_matrix,
                                    dist_coeffs)
Rmat2, jacobian2 = cv2.Rodrigues(rvec2)


# retvl1, rvec1, tvec1 = cv2.solvePnP(building,
#                                     bld_img1,
#                                     oldcammtx,
#                                     olddistcoeff)
# Rmat1, jacobian1 = cv2.Rodrigues(rvec1)
# retvl2, rvec2, tvec2 = cv2.solvePnP(building,
#                                     bld_img1,
#                                     oldcammtx,
#                                     olddistcoeff)
# Rmat2, jacobian2 = cv2.Rodrigues(rvec2)

P1 = np.dot(camera_intrinsic_matrix, np.hstack((Rmat1, tvec1)))
P2 = np.dot(camera_intrinsic_matrix, np.hstack((Rmat2, tvec2)))

E1 ,mask0 = cv2.findEssentialMat(undist1, undist2, camera_intrinsic_matrix)
F1 = np.dot(np.linalg.inv(camera_intrinsic_matrix).T, np.dot(E1, np.linalg.inv(camera_intrinsic_matrix)))
F2,_ = cv2.findFundamentalMat(undist1, undist2)
refined001, refined600 = cv2.correctMatches(F1, undist1.reshape(1,-1,2), undist2.reshape(1,-1,2))
#
refined001 = refined001.reshape(-1,1,2)
refined600 = refined600.reshape(-1,1,2)

X3d = cv2.triangulatePoints(P1, P2, undist1, undist2)
X3d = from_homog(X3d)

P1_reproj, jacob1 = cv2.projectPoints(building, rvec1, tvec1, camera_intrinsic_matrix, dist_coeffs)
P2_reproj, jacob2 = cv2.projectPoints(building, rvec2, tvec2, camera_intrinsic_matrix, dist_coeffs)

# P1_reproj, jacob1 = cv2.projectPoints(building, rvec1, tvec1, oldcammtx, olddistcoeff)
# P2_reproj, jacob2 = cv2.projectPoints(building, rvec2, tvec2, oldcammtx, olddistcoeff)

print abs(P1_reproj-undist1)[...,0].mean()
print abs(P1_reproj-undist1)[...,1].mean()
print np.linalg.norm(P1_reproj-undist1, axis=2).mean()

# print abs(P2_reproj-undist2)[...,0].mean()
# print abs(P2_reproj-undist2)[...,1].mean()

print abs(X3d.T.reshape(-1,1,3)-building)[...,0].mean()
print abs(X3d.T.reshape(-1,1,3)-building)[...,1].mean()
print abs(X3d.T.reshape(-1,1,3)-building)[...,2].mean()
print np.linalg.norm((X3d.T.reshape(-1,1,3)-building), axis=2).mean()
v
# plt.scatter(P2_reproj[...,0],-P2_reproj[...,1])
plt.scatter(undist1[...,0], 2160-undist1[...,1],  label='Original corners', s=15)
plt.scatter(P1_reproj[...,0],2160-P1_reproj[...,1], label='Reprejected corners', s=15)
plt.xlabel('x (pixel)')
plt.ylabel('y (pixel)')
plt.legend()
# plt.scatter(undist2[...,0], -undist2[...,1])
plt.show()
fdsa

vdfv

# with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/3DPointCloud40.txt", "w") as f0:
#     for i in range(X3d.shape[1]):
#         for j in range(3):
#             f0.write(str(X3d[j,i]) + '\t')
#         f0.write("\n")
# with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/im1_bldpts.txt", "w") as f1:
#     for i in range(P1_reproj.shape[0]):
#         for j in range(2):
#             f1.write(str(P1_reproj[i,:,j].squeeze()) + '\t')
#         f1.write("\n")

# with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/im2_bldpts.txt", "w") as f1:
#     for i in range(P2_reproj.shape[0]):
#         for j in range(2):
#             f1.write(str(P2_reproj[i,:,j].squeeze()) + '\t')
        # f1.write("\n")

# Default params
# int     nfeatures = 0,
# int     nOctaveLayers = 3,
# double  contrastThreshold = 0.04,
# double  edgeThreshold = 10,
# double  sigma = 1.6 
sift_params = dict(nfeatures = 0,nOctaveLayers = 3,contrastThreshold = 0.03,edgeThreshold = 8,sigma = 1.6)
sift = cv2.xfeatures2d.SIFT_create(**sift_params)

bg_file1 = '/Users/yujieli/Documents/CFS_Video_Analysis-master/NFramesRaw5/frame270.png'
img1 = cv2.imread(bg_file1,0)
img1_cut = getbg(img1,mask)
img1_cut = cv2.undistort(img1_cut, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/undisttest1.png', img1)
kp1, des1 = sift.detectAndCompute(img1_cut, None)

bg_file2 = '/Volumes/Menelaus/EQ7_North/frame1649.png'
img2 = cv2.imread(bg_file2,0)
img2_cut = getbg(img2,mask7)
img2_cut = cv2.undistort(img2_cut, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/undisttest2.png', img2)
kp2, des2 = sift.detectAndCompute(img2_cut, None)



bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_ini = [] 
for m,n in matches:
    if m.distance < 0.82*n.distance:
        good_ini.append(m)

srcpts_ini = np.float32([ kp1[m.queryIdx].pt for m in good_ini ]).reshape(-1,1,2)
dstpts_ini = np.float32([ kp2[m.trainIdx].pt for m in good_ini ]).reshape(-1,1,2)
des_ini = np.float32([des1[m.queryIdx] for m in good_ini])
des_ini2 = np.float32([des2[m.trainIdx] for m in good_ini])

# M, mask0 = cv2.findHomography(srcpts_ini, dstpts_ini, cv2.RANSAC,5.0)
# E ,mask1 = cv2.findEssentialMat(srcpts_ini, dstpts_ini, camera_intrinsic_matrix)

# F = np.dot(np.linalg.inv(camera_intrinsic_matrix).T, np.dot(E, np.linalg.inv(camera_intrinsic_matrix)))
F, mask0 = cv2.findFundamentalMat(srcpts_ini, dstpts_ini)
# print F
refined_src, refined_dst = cv2.correctMatches(F, srcpts_ini[(mask0>0).squeeze()].reshape(1,-1,2),
                                              dstpts_ini[(mask0>0).squeeze()].reshape(1,-1,2))
matchesMask = mask0.ravel().tolist()

refined_src = refined_src.reshape(-1,1,2)
refined_dst = refined_dst.reshape(-1,1,2)
des_ini = des_ini[(mask0>0).squeeze()]
des_ini2 = des_ini2[(mask0>0).squeeze()]
print 'Max reproj error:',(abs(refined_src-srcpts_ini[(mask0>0).squeeze()]).max())
print 'Mean reproj error:', (abs(refined_dst-dstpts_ini[(mask0>0).squeeze()]).mean())
# retval, Rmat, tvec, mask1 = cv2.recoverPose(E, refined_src, refined_dst, camera_intrinsic_matrix, 50)

# P1p = np.dot(camera_intrinsic_matrix, np.eye(3,4))
# P2p = np.dot(camera_intrinsic_matrix, np.hstack((Rmat, tvec.reshape(-1,1))))
print "Initial pairs", np.nonzero(mask0)[0].shape[0]

draw_params = dict(
                    singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                flags = 2)

img3 = cv2.drawMatches(img1_cut,kp1,img2_cut,kp2,good_ini,None, **draw_params)

plt.imshow(img3, cmap='gray')
plt.show()
cs

# Construct first 3D point set
X3d = cv2.triangulatePoints(P1, P2, refined_src, refined_dst)
X3d = from_homog(X3d)
# hightmask = np.nonzero(X3d[1] < 20000.)[0]
# X3d = X3d[:, hightmask]
# des_ini = des_ini[hightmask]
# des_ini2 = des_ini2[hightmask]
# refined_src = refined_src[hightmask]
# refined_dst = refined_dst[hightmask]

with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/3DBackGround.txt", "w") as f0:
    for i in range(X3d.shape[1]):
        for j in range(3):
            f0.write(str(X3d[j,i]) + '\t')
        f0.write("\n")

with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/im1_BackGround.txt", "w") as f1:
    for i in range(refined_src.shape[0]):
        for j in range(2):
            f1.write(str(refined_src[i,:,j].squeeze()) + '\t')
        f1.write("\n")

with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/test/im2_BackGround.txt", "w") as f1:
    for i in range(refined_dst.shape[0]):
        for j in range(2):
            f1.write(str(refined_dst[i,:,j].squeeze()) + '\t')
        f1.write("\n")
dfcghv
with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/CamPoseUndist_EQ7.txt", "w") as f:
    frame_i = 1650
    while(frame_i <= 2888):

        print frame_i

        img2 = cv2.imread(
            "/Volumes/Menelaus/EQ7_North/frame{:d}.png".format(frame_i),0)
        img2_Cut = getbg(img2,mask7)
        img2_Cut = cv2.undistort(img2_Cut, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
        # plt.plot(img2_Cut)
        # plt.show()
        kp3, des3 = sift.detectAndCompute(img2_Cut, None)
        print "KP detected:", len(kp3)
        # matches = bf.knnMatch(des_ini, des3, k=2)
        matches2 = bf.knnMatch(des_ini2, des3, k=2)
        good = []
        for m,n in matches2:
            if m.distance < 0.82 * n.distance:
                good.append(m)

        print "Good pairs:", len(good)

        # draw_params = dict(
        #             singlePointColor = None,
        #            matchesMask = None, # draw only inliers
        #         flags = 2)

        # img3 = cv2.drawMatches(img2_cut,kp2,img2_Cut,kp3,good,None,**draw_params)

        # plt.imshow(img3, cmap='gray'),plt.show()


        dst_pts_3d = np.float32([X3d[:, m.queryIdx] for m in good]).reshape(-1,1,3)
        src_pts = np.float32([kp3[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        dist_coeffs = np.zeros((4,1))
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(dst_pts_3d, 
                                                         src_pts, 
                                                         camera_intrinsic_matrix, 
                                                         dist_coeffs)

        Rmat, jacobian = cv2.Rodrigues(rvec)

        f.write(str(frame_i) + "\t")
        for i in range(3):
            f.write(str(rvec.squeeze()[i]) + "\t")
        for i in range(3):
            f.write(str(tvec.squeeze()[i]) + "\t")
        f.write("\n")

        with open("/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/Matches_Cam{:d}.txt".format(frame_i), "w") as ff:
            for i in range(len(good)):
                frameMatch = good[i]
                ff.write(str(frameMatch.queryIdx) + '\t')
                ff.write(str(kp3[frameMatch.trainIdx].pt[0]) + '\t')
                ff.write(str(kp3[frameMatch.trainIdx].pt[1]) + '\t')
                ff.write("\n")

        frame_i += 1


# dst_pts_homog = np.vstack((dst_pts_3d.reshape(-1,3).T, np.ones((1,len(good)))))
# P3 = np.dot(camera_intrinsic_matrix,np.hstack((Rmat, tvec)))
# X33d = np.dot(P3,dst_pts_homog)
# X3 = from_homog(X33d)
# # print X3.T[inliers]
# # print src_pts.reshape(-1,2)[inliers]
# print (X3.T[inliers]-src_pts.reshape(-1,2)[inliers]).mean()
# print (X3.T[inliers]-src_pts.reshape(-1,2)[inliers]).max()
