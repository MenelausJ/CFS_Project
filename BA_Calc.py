import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm
import copy
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def to_homog(points):
    if np.ndim(points) == 1:
        pointsh = np.hstack((points,1))
        return pointsh / norm(pointsh)
    else:
        pointsh = np.vstack((points,np.ones(points.shape[1])))
        return pointsh / norm(pointsh, axis=0)


def from_homog(points_homog):
    return points_homog[:-1]/points_homog[-1]


def parameterize(X):
    X = X / norm(X)
    a_param = X[0]
    b_param = X[1:]
    v_param = 2/np.sinc(np.arccos(a_param)/np.pi) * b_param
    if norm(v_param) > np.pi:
        v_param = (1-2*np.pi/norm(v_param)*np.ceil((norm(v_param)-np.pi)\
                                                    /(2*np.pi)))* v_param

    return v_param


def deparameterize(v_param):
    if norm(v_param) > np.pi:
        v_param = (1-2*np.pi/norm(v_param)*np.ceil((norm(v_param)-np.pi)\
                                                    /(2*np.pi)))* v_param
    X = np.hstack((np.cos(norm(v_param/2)),
                    np.sinc(1/np.pi*norm(v_param/2))/2 * v_param))

    return X


def cross2mat(w):

    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=np.float64)


def d_param(Xvec):
#     Calculate the Jacobian of homogeneous vector Xvec WRT its parameterization.
    dim = len(Xvec)
    Xvec  = Xvec / norm(Xvec)
    a_param = Xvec[0]
    b_param = Xvec[1:]
    v_param = parameterize(Xvec).reshape(-1,1)
    Jcb_p = np.zeros((dim, dim-1))
    Jcb_p[0] = -0.5 * b_param.T
    Jcb_p[1:] = np.sinc(1/np.pi*norm(v_param/2))/2 * np.eye(dim-1) +\
    1/(4*norm(v_param)) *\
    (np.cos(norm(v_param/2)) / (norm(v_param/2)) -\
     np.sin(norm(v_param/2)) / norm(v_param/2)**2) *\
    np.dot(v_param, v_param.T)

    return Jcb_p


def R2w(R):
    # given a rotation matrix R return the angle-axis representation
    u, s, v = np.linalg.svd(R - np.eye(3))
    v = v[np.argmin(abs(s))]
    vh = np.array([[R[2,1]-R[1,2]], [R[0,2]-R[2,0]], [R[1,0]-R[0,1]]], dtype=np.float64)
    sin = 1/2 * np.dot(v, vh)
    cos = (np.trace(R) - 1) / 2
    theta = np.arctan2(sin, cos)
    return v * theta


def w2R(w):
    # given the angle-axis representation w return the rotation matrix
    thta = norm(w)
    R = np.eye(3) * np.cos(thta) +\
    np.sinc(thta / np.pi) * np.array([[0, -w[2], w[1]],
                                       [w[2], 0, -w[0]],
                                       [-w[1], w[0], 0]], dtype=np.float64) +\
    (1 - np.cos(thta)) / thta**2 * np.dot(w.reshape(-1,1), w.reshape(1,-1))
    return R

frame_0 = 1650
frame_end = 2050
intrinsic_K = np.float64([[2.21891528e+03,  0.00000000e+00,   1.93571064e+03],
                          [0.00000000e+00,  2.26785645e+03,   1.05642772e+03],
                          [0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])
inv_K = LA.inv(intrinsic_K)

scenePts = []

with open('/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/3DBackGround.txt', 'r') as f0:
    while 1:
            line = f0.readline()
            if not line:
                break

            line = line[:line.find('\t\n')]
            X, Y, Z = [float(i) for i in line.split('\t')]
            # X, Y, Z = [float(i) for i in line.split('\t')] + 500.0 * np.random.rand(3)
            scenePts.append(np.array([X, Y, Z], dtype=np.float64))

np.savetxt("/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/3DBG_Raw.txt",\
 np.asarray(scenePts, dtype=np.float64))

numPts = len(scenePts)

scenePtsParamed = []
for i in range(numPts):
    scenePtsParamed.append(parameterize(to_homog(scenePts[i])))

camPoses = []
with open('/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/CamPoses_EQ7.txt', 'r') as f1:
    while 1:
            line = f1.readline()
            if not line:
                break

            line = line[:line.find('\t\n')]
            frame, rx, ry, rz, X, Y, Z = [float(i) for i in line.split('\t')]
            # Only choose frames every 1/3 seconds
            if frame > frame_end:
                break
            if (frame - frame_0) % 10 > 0:
                continue
            else:
                camPoses.append(np.array([frame, rx, ry, rz, X, Y, Z], dtype=np.float64))


numCams = len(camPoses)
print 'Number of frames used:', numCams

ptsCaptured = []
for i in range(numCams):
    ptsCaptured.append(np.loadtxt('/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/Matches_Cam{:d}.txt'
                             .format(frame_0 + i * 10)))


# Parameters to tune:
lmb = 3000.
stepLengthChange = 10
maxTrials = 10
maxIters = 10
costThresh = 1e-10 # Convergence criteria
stepThresh = 1e+16 # Constrain Lambda
errorThresh = 4e-4 # For Huber penalty kernel
retval = 0

# ************************************************************************************
# ***************************** LM algorithm starts here *****************************
# ************************************************************************************


for iters in range(maxIters):

    if lmb >= stepThresh:
        print 'Min step length reached'
        break

    if retval >= 2:
        print 'Congratulations! Converged!'
        break

    print 'Iteration #', iters + 1
    # Initializing
    cost_ini = 0
    eps_a = np.zeros((6*numCams), dtype=np.float64)
    eps_b = np.zeros((3*numPts), dtype=np.float64)
    count = 0
    frame_i = frame_0 + 0
    block_U = [np.zeros((6,6), dtype=np.float64) for i in range(numCams)]
    block_V = [np.zeros((3,3), dtype=np.float64) for i in range(numPts)]
    block_W = [[np.zeros((6,3), dtype=np.float64) for j in range(numPts)] for i in range(numCams)]
    # Hes = np.zeros((6*numCams+3*numPts, 6*numCams+3*numPts))

    # Jacobians and Hessian blocks
    while frame_i <= frame_end:
        # Search within each camera
        # print frame_i
        frame, rx, ry, rz, X, Y, Z = camPoses[count]
        frame = int(frame)
        if frame != frame_i:
            print 'Inconsistent matching'
            break

        rVec = np.array([rx, ry, rz], dtype=np.float64)
        rotMat = w2R(rVec)
        tVec = np.array([X, Y, Z], dtype=np.float64)
        theta = norm(rVec)
        rVec_x = cross2mat(rVec)
        projMat = np.hstack((rotMat, tVec.reshape(-1,1)))
        scene_i = ptsCaptured[count]
        # numDetected = len(scene_i)
        # print 'Matches detected:', numDetected
        maxError = 0
        minError = 100

        for ind, x, y in scene_i:
            # Search within each point match
            ind = int(ind)
            xhat, yhat = from_homog(np.dot(inv_K, np.array([x, y, 1], dtype=np.float64)))
            worldX = scenePts[ind]
            worldX_x = cross2mat(worldX)
            camX = np.dot(rotMat, worldX) + tVec
            eps_ij = np.array([xhat, yhat]) - from_homog(camX)
            errorRaw = np.dot(eps_ij, eps_ij)
            # xReproj, yReproj = from_homog(camX)
            # Perhaps I should use calculated image points instead of given points here
            A1 = np.array([[1/camX[2], 0, -xhat/camX[2]],
                       [0, 1/camX[2], -yhat/camX[2]]])
            # A1 = np.array([[1/camX[2], 0, -xReproj/camX[2]],
            #            [0, 1/camX[2], -yReproj/camX[2]]])
            A2 = -np.sinc(theta / np.pi) * worldX_x +\
            np.dot(np.cross(rVec, worldX).reshape(-1,1), rVec.reshape(1,-1)) / theta *\
            (theta * np.cos(theta) - np.sin(theta)) / theta**2 +\
            np.dot(np.cross(rVec, np.cross(rVec, worldX)).reshape(-1,1), rVec.reshape(1,-1)) / theta *\
            (theta * np.sin(theta) - 2 * (1 - np.cos(theta))) / theta**3 +\
            ((1 - np.cos(theta)) / theta**2) * (-np.dot(rVec_x, worldX_x) +
                                                          cross2mat(-np.cross(rVec, worldX)))

            B2 = d_param(to_homog(worldX))

            jacobian1 = np.hstack((np.dot(A1, A2), A1))
            jacobian2 = np.dot(A1, np.dot(projMat, B2))
            block_U[count] += np.dot(jacobian1.T, jacobian1)
            block_V[ind] += np.dot(jacobian2.T, jacobian2)
            block_W[count][ind] = np.dot(jacobian1.T, jacobian2)


            # The original sparse Hessian
            # Hes[6*count:6*count+6, 6*count:6*count+6] += np.dot(jacobian1.T, jacobian1)
            # Hes[(6*numCams+3*ind):(6*numCams+3*ind+3), 6*count:6*count+6] = np.dot(jacobian2.T, jacobian1)
            # Hes[6*count:6*count+6, (6*numCams+3*ind):(6*numCams+3*ind+3)] = np.dot(jacobian1.T, jacobian2)
            # Hes[(6*numCams+3*ind):(6*numCams+3*ind+3), (6*numCams+3*ind):(6*numCams+3*ind+3)] += np.dot(jacobian2.T, jacobian2)

            # print errorRaw
            if maxError <= errorRaw:
                maxError = errorRaw + 0
            if minError >= errorRaw:
                minError = errorRaw + 0

            cost_ini += errorRaw

            # Count is the index of frames while ind the index of points
            eps_a[6*count:6*count+6] += np.dot(jacobian1.T, eps_ij)
            eps_b[3*ind:3*ind+3] += np.dot(jacobian2.T, eps_ij)

            
        frame_i += 10
        count += 1

    print 'Initial cost:', cost_ini
    print 'Max error:', np.sqrt(maxError)
    print 'Min error:', np.sqrt(minError)


    # LM trials
    for trials in range(maxTrials):
        # print 'Trial:', trials + 1
        camPoses_new = copy.deepcopy(camPoses)
        scenePtsParamed_new = copy.deepcopy(scenePtsParamed)
        mat_S = np.zeros((6*numCams, 6*numCams))
        block_V_inv_s = [np.zeros((3,3)) for i in range(numPts)]
        for j in range(numCams):
            mat_S[6*j:6*j+6, 6*j:6*j+6] += block_U[j] + lmb*np.eye(6)
            for k in range(numCams):
                for i in range(numPts):
                    block_V_inv_s[i] = LA.inv(block_V[i] + lmb*np.eye(3))
                    mat_S[6*j:6*j+6, 6*k:6*k+6] -= np.dot(np.dot(block_W[j][i], block_V_inv_s[i]),
                                                          block_W[k][i].T)

        # The correlation matrix describes how overlapped observed scenePts are for all frames 
        # Comment out U blocks for visual effect
        # mat_Corr = np.zeros((numCams, numCams))
        # for j in range(numCams):
        #     for k in range(numCams):
        #         mat_Corr[j, k] = np.mean(mat_S[6*j:6*j+6, 6*k:6*k+6])
        # plt.figure(figsize=(5, 5))
        # plt.imshow(mat_Corr)
        # plt.show()
        # Stop!
        epsilon = eps_a.copy()
        for j in range(numCams):
            for i in range(numPts):
                epsilon[6*j:6*j+6] -= np.dot(np.dot(block_W[j][i], block_V_inv_s[i]),
                                             eps_b[3*i:3*i+3])

        delta_a = LA.lstsq(mat_S, epsilon)[0]
        delta_b = np.zeros((3*numPts))
        for i in range(numPts):
            delta_b[3*i:3*i+3] = np.dot(block_V_inv_s[i], eps_b[3*i:3*i+3])
            for j in range(numCams):
                delta_b[3*i:3*i+3] -= np.dot(block_V_inv_s[i], np.dot(block_W[j][i].T, delta_a[6*j:6*j+6]))
        # Direct calculation without Schur complement, just to debug
        # The result is stack of delta_a and delta_b
        # delta = LA.lstsq(Hes+lmb*np.eye(6*numCams+3*numPts), np.hstack((eps_a, eps_b)))[0]

        for i in range(numCams):
            camPoses_new[i][1:] += delta_a[6*i:6*i+6]

        for i in range(numPts):
            scenePtsParamed_new[i] += delta_b[3*i:3*i+3]

        scenePts_new = []
        for i in range(numPts):
            scenePts_new.append(from_homog(deparameterize(scenePtsParamed_new[i])))

        frame_j = frame_0 + 0
        cost_new = 0
        count_new = 0

        while frame_j <= frame_end:
            # Search within each camera
            scene_i = ptsCaptured[count_new]
            frame, rx, ry, rz, X, Y, Z = camPoses_new[count_new]
            rotMat = w2R(np.array([rx, ry, rz], dtype=np.float64))
            # rotMat, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
            tVec = np.array([X, Y, Z], dtype=np.float64)

            for ind, x, y in scene_i:
                ind = int(ind)
                xhat, yhat = from_homog(np.dot(inv_K, np.array([x, y, 1], dtype=np.float64)))
                worldX = scenePts_new[ind]
                camX = np.dot(rotMat, worldX) + tVec
                eps_ij = np.array([xhat, yhat]) - from_homog(camX)
                cost_new += np.dot(eps_ij, eps_ij)

            frame_j += 10
            count_new += 1

        # print'Trial cost:', cost_new
        if cost_new <= cost_ini:
            print 'Updated!'
            print 'Lambda:', lmb
            print 'Interior trials', trials + 1
            lmb = lmb / stepLengthChange
            camPoses = copy.deepcopy(camPoses_new)
            scenePtsParamed = copy.deepcopy(scenePtsParamed_new)
            scenePts = copy.deepcopy(scenePts_new)

            if cost_ini - cost_new <= costThresh and trials >= 3:
                retval += 1
            break

        else:
            # print 'Increase lambda'
            lmb = lmb * stepLengthChange

np.savetxt("/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/3DBG_New.txt",\
 np.asarray(scenePts, dtype=np.float64))
print camPoses[0]
