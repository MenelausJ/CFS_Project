# RBG filter for feature points
import sys
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


oldcammtx = np.float32([[2.32729520e+03, 0.00000000e+00, 1.92959380e+03],
                        [0.00000000e+00, 2.32850245e+03, 1.05881793e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
olddistcoeff = np.float32([-0.14374176,  0.13863309, -0.00090426,  0.00109337, -0.04210762])

camera_intrinsic_matrix = np.float32([[2.21891528e+03,  0.00000000e+00,   1.93571064e+03],
                                      [0.00000000e+00,  2.26785645e+03,   1.05642772e+03],
                                      [0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])
dist_coeffs = np.zeros((4,1), np.float32)


mask_l = 2420
mask_r = 2560
nmsk = np.zeros((2160, 3840), dtype='uint8')
nmsk[380:-260,mask_l:mask_r] = 1
nmsk = cv2.undistort(nmsk, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
# cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/FtPtsUndist/MASK.png', nmsk)
dltkernel = np.ones([5,5])

def findMarkers_EQ7(frame_ori, mask, dltknl = None):
	# Find all six markers with adaptive mask.

	oldcammtx = np.float32([[2.32729520e+03, 0.00000000e+00, 1.92959380e+03],
                        	[0.00000000e+00, 2.32850245e+03, 1.05881793e+03],
                        	[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
	olddistcoeff = np.float32([-0.14374176,  0.13863309, -0.00090426,  0.00109337, -0.04210762])
	camera_intrinsic_matrix = np.float32([[2.21891528e+03,  0.00000000e+00,   1.93571064e+03],
                                      	[0.00000000e+00,  2.26785645e+03,   1.05642772e+03],
                                      	[0.00000000e+00,  0.00000000e+00,   1.00000000e+00]])
	dist_coeffs = np.zeros((4,1), np.float32)

	frame = cv2.undistort(frame_ori, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
	frame = cv2.bitwise_and(frame_ori, frame_ori, mask=nmsk)
	frame0 = frame.copy()
	frame = gaussian_filter(frame,2.)
	frame[frame0[...,0] < 60] = 0
	if dltknl is not None:
		frame = cv2.dilate(frame, dltknl)

	lower_white = np.array([82,82,82])
	upper_white = np.array([150,150,150])
	maskflt = cv2.inRange(frame, lower_white, upper_white)

	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	frame = cv2.bitwise_and(frame, frame, mask=maskflt)
		
	frame[frame<35]=0
	frame[frame>0]=255
	_1, contours, _2 = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	markers = []
	for i in range(len(contours)):
		cnt = contours[i]
		x, y, w, h = cv2.boundingRect(cnt)
		if w>11 and w<40 and h>10 and h<32:
			patch = frame[y:y+h, x:x+w]
 			if len(np.nonzero(patch>0)[0])> 0.65 * w * h:
				markers.append(np.array([x+1, y+1]))
				# frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)

	return markers, frame


frame_i = 1650+1235
with open('/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ7/RawDispUndist7_EQ7.txt', 'w') as f:
	while (frame_i <= 1650+1238):
		frame_ori = cv2.imread('/Volumes/Menelaus/EQ7_North/frame%d.png'%(frame_i))
		fpts, frame = findMarkers_EQ7(frame_ori, nmsk)
		if len(fpts) != 6:
			print 'Dilate'
			fpts, frame = findMarkers_EQ7(frame_ori, nmsk, dltkernel)
		if len(fpts) != 6:
			print 'Mask moved left'
			mask_l -= 20
			mask_r -= 20
			nmsk = np.zeros((2160, 3840), dtype='uint8')
			nmsk[380:-260,mask_l:mask_r] = 1
			nmsk = cv2.undistort(nmsk, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
			fpts, frame = findMarkers_EQ7(frame_ori, nmsk)
			if len(fpts) != 6:
				print 'Mask moved right'
				mask_l += 40
				mask_r += 40
				nmsk = np.zeros((2160, 3840), dtype='uint8')
				nmsk[380:-260,mask_l:mask_r] = 1
				nmsk = cv2.undistort(nmsk, oldcammtx, olddistcoeff, None, camera_intrinsic_matrix)
				fpts, frame = findMarkers_EQ7(frame_ori, nmsk)
				if len(fpts) != 6:
					print 'RGB filter failed, modify params'
					print 'Features tracked:', len(fpts)
					plt.imshow(frame)
					plt.show()
					break

		fpts = sorted(fpts, key = lambda x:x[1])

		f.write(str(frame_i) + "\t")
		for i in range(len(fpts)):	
			f.write(str(fpts[i][0]) + "\t")
			f.write(str(fpts[i][1]) + "\t")
			
		f.write("\n")

		# cv2.imwrite("/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ6ft/feat%d.png"%(frame_i), frame)
		
		print frame_i
		frame_i += 1
