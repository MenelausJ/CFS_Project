import numpy as np
import cv2

oldcammtx = np.float32([[2.32729520e+03, 0.00000000e+00, 1.92959380e+03],
                        [0.00000000e+00, 2.32850245e+03, 1.05881793e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
olddistcoeff = np.float32([-0.14374176,  0.13863309, -0.00090426,  0.00109337, -0.04210762])

camera_intrinsic_matrix = np.float32([[2.21891528e+03,  0.00000000e+00,   1.93571064e+03],
                                      [0.00000000e+00,  2.26785645e+03,   1.05642772e+03],
                                      [0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])
dist_coeffs = np.zeros((4,1), np.float32)

filename1 = "/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/CamPoses_EQ7.txt"

Rvecs = []
tvecs = []
with open(filename1, "r") as f:

    while 1:
        line = f.readline()
        if not line:
            break
        line = line[:line.find('\t\n')]
        frame,r1,r2,r3,t1,t2,t3 = [float(i) for i in line.split('\t')]
        Rvecs.append(np.float32([r1,r2,r3]))
        tvecs.append(np.float32([t1,t2,t3]))
        if frame >= 2888:
            break

# xleft = +2325
# xright = 2460
# ytop = +620
# ybottom = 1700

# Top wall: 1195mm
filename2 = "/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/RawDisps_EQ7.txt"

pts = []
with open(filename2, "r") as f2:
    while 1:
        line = f2.readline()
        if not line:
            break
        line = line[:line.find('\t\n')]
        frame,p1,p2,p3,p4,p5,p6,\
            p7,p8,p9,p10,p11,p12 = [float(i) for i in line.split('\t')]

        # pts.append(np.float32([p11+2325,p12+620,p9+2325,p10+620,p7+2325,p8+620,
        #                        p5+2325,p6+620,p3+2325,p4+620,p1+2325,p2+620]))
        pts.append(np.float32([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]))




filename3 = "/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/StoryDriftZ.txt"
frm = 1650
with open(filename3, "w") as f3:
    Z = 30000.
    X_disp = np.zeros((len(pts), 6))
    for i in range(len(pts)):

        f3.write(str(frm) + "\t")

        R, jacobian = cv2.Rodrigues(Rvecs[i])
        KR = np.dot(camera_intrinsic_matrix, R)
        tp = np.dot(camera_intrinsic_matrix, tvecs[i].reshape(-1,1))
        A = np.linalg.inv(KR)
        pt = pts[i]



        for story in range(6):
            x, y = pt[2*story:2*story+2]
            xtilda = np.float32([x, y, 1]).reshape(-1,1)
            Y = story * 3050. + 1195. + 1056.


            w = (Z + np.dot(A[-1], tp)) / np.dot(A[-1], xtilda)
            wp = (Y + np.dot(A[1], tp)) / np.dot(A[1], xtilda)
            # Fix Y
            # Xvec = np.dot(A, (wp*xtilda - tp))
            # Fix Z
            Xvec = np.dot(A, (w*xtilda - tp))
            X_disp[i, story] = Xvec[0]
            f3.write(str(Xvec[0].squeeze()) + '\t')
            f3.write(str(Xvec[1].squeeze()) + '\t')
            f3.write(str(Xvec[2].squeeze()) + '\t')

        f3.write('\n')
        frm += 1
        if frm >= 2888:
            break
X_disp_smooth = np.zeros((len(pts), 6))
for i in range(6):
    rft = np.fft.rfft(X_disp[:,i])
    rft[60:] = 0
    X_disp_smooth[:-1,i] = np.fft.irfft(rft)
    X_disp_smooth[-1,i] = X_disp_smooth[-2,i]

filename4 = "/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/StoryDriftZSmooth.txt"
with open(filename4, "w") as f4:
    frm = 1650
    for i in range(len(pts)):
        f4.write(str(frm) + "\t")
        for story in range(6):
            f4.write(str(X_disp_smooth[i, story].squeeze()) + '\t')

        f4.write('\n')
        frm += 1
        if frm >= 2888:
            break




