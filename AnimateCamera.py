import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from matplotlib import pyplot as plt
import utilities as ut
from utilities import Arrow3D
# import global_figure as gf

camera_poseX = []
camera_poseY = []
camera_poseZ = []
camera_orientationX = []
camera_orientationY = []
camera_orientationZ = []

camera_pose = [0,0,0]
camera_orientation = [0,0,0]

filename = '/Users/yujieli/Documents/CFS_Video_Analysis-master/test/CamPoseUndist_63.txt'

flag = 1

imageScale = 0.1

pattern_image = cv2.imread('/Users/yujieli/Documents/CFS_Video_Analysis-master/test/bldrect.png')
pattern_image = cv2.resize(pattern_image, None, fx = imageScale, fy=imageScale)

# pattern_image = cv2.flip(cv2.transpose(pattern_image), 0)

# Plot pattern
height = pattern_image.shape[0]
width = pattern_image.shape[1]
yy, zz = np.meshgrid(np.linspace(0, width/imageScale, width), 
    np.linspace(height/imageScale,0, height))

X = np.zeros_like(zz)
Y = yy
Z = zz


pixel2meter = 0.011
meter2inch = 39.3701
pixel2inch = pixel2meter * meter2inch

with open(filename, "r") as f:
    while 1:
        line = f.readline() 
        if not line:
            break
            pass

        line = line[:line.find('\t\n')]
        frame_i, camera_orientation[0], camera_orientation[1], camera_orientation[2], \
        camera_pose[0], camera_pose[1], camera_pose[2] \
        = [float(i) for i in line.split('\t')]
        rvec = np.float32([camera_orientation[0], camera_orientation[1], camera_orientation[2]])
        tvec = np.float32([camera_pose[0], camera_pose[1], camera_pose[2]])
        Rmat, _ = cv2.Rodrigues(rvec)
        tvecp = -np.dot(Rmat.T, tvec) * imageScale
        orientation = np.dot(Rmat.T, np.float32([0, 0, 1000]))
        # print orientation

        if flag:
            frame_0 = int(frame_i)
            flag = 0

        camera_poseX.append(tvecp[2] - 30000.*imageScale)
        camera_poseY.append(tvecp[0])
        camera_poseZ.append(tvecp[1] + 19579.*imageScale)
        camera_orientationX.append(orientation[2])
        camera_orientationY.append(orientation[0])
        camera_orientationZ.append(orientation[1])

# print max(camera_poseX),max(camera_poseY),max(camera_poseZ)
# print min(camera_poseX),min(camera_poseY),min(camera_poseZ)

fig = plt.figure()

frames = len(camera_poseX)

# ax1 = fig.add_subplot(321, xlim=(0, frames/30.), ylim=(-5000, 5000)) 
# ax2 = fig.add_subplot(323, xlim=(0, frames/30.), ylim=(-5000, 5000))
# ax3 = fig.add_subplot(325, xlim=(0, frames/30.), ylim=(-5000, 5000))

# ax1.set_title('X Relative Displacement',fontweight="bold", size=6) # Title
# ax1.set_ylabel('X(meter)', fontsize = 6) # Y label
# ax1.set_xlabel('time(sec)', fontsize = 6) # X label
# line1, = ax1.plot([], [], lw=1, color = (0,0,1))   
# ax1.legend(loc='upper right')
# text1 = ax1.text(1, 20, "0", fontsize=6)

# ax2.set_title('Y Relative Displacement',fontweight="bold", size=6) # Title
# ax2.set_ylabel('Y(meter)', fontsize = 6) # Y label
# ax2.set_xlabel('time(sec)', fontsize = 6) # X label
# line2, = ax2.plot([], [], lw=1, color = (0,0,1))   
# ax2.legend(loc='upper right')
# text2 = ax2.text(1, 20, "0", fontsize=6)

# ax3.set_title('Z Relative Displacement',fontweight="bold", size=6) # Title
# ax3.set_ylabel('Z(meter)', fontsize = 6) # Y label
# ax3.set_xlabel('time(sec)', fontsize = 6) # X label
# line3, = ax3.plot([], [], lw=1, color = (0,0,1))   
# ax3.legend(loc='upper right')
# text3 = ax3.text(1, 20, "0", fontsize=6)

ax = fig.add_subplot(111, projection='3d')
# Set axes labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(0, 3600)
ax.set_ylim3d(0, 1500)
ax.set_zlim3d(0, 2000)

plt.gca().set_aspect('equal', adjustable=None)

ax_image = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=pattern_image / 255., shade=False)


# pattern_image = cv2.imread('/Users/yujieli/Documents/CFS_Video_Analysis-master/test/refgen.png')
# pattern_image = cv2.resize(pattern_image, None, fx = imageScale, fy=imageScale)

# # pattern_image = cv2.flip(cv2.transpose(pattern_image), 0)

# offSetX = 276. * pixel2meter
# offSetY = 988. * pixel2meter
# # Plot pattern
# height = pattern_image.shape[0]
# width = pattern_image.shape[1]
# xx, yy = np.meshgrid(np.linspace(offSetX, offSetX + width* (pixel2meter/imageScale) , width), 
#     np.linspace(height* (pixel2meter/imageScale) + offSetY,offSetY, height))

# X = xx
# Y = yy
# Z = 20.*np.ones(xx.shape)

# ax_image2 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=pattern_image / 255., shade=False)


plt.tight_layout(w_pad=2.0, h_pad=2.0)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)

flag = 0

with writer.saving(fig, '/Users/yujieli/Documents/CFS_Video_Analysis-master/EQ6/testanim.MP4', dpi = 200):
    x = np.linspace(0, frames/30., frames)
    for i in range(frames):    
        camera_pose[0] = camera_poseX[i]
        camera_pose[1] = camera_poseY[i]
        camera_pose[2] = camera_poseZ[i]
        camera_orientation[0] = camera_orientationX[i]
        camera_orientation[1] = camera_orientationY[i]
        camera_orientation[2] = camera_orientationZ[i]


        if flag:
            # gf.ax.texts.remove(ax_text)
            ax.collections.remove(ax_camera)
            ax.artists.remove(arrow)

        camera_pose = [k for k in camera_pose]
        camera_pose[0] =  -camera_pose[0]
        # print camera_pose
        # print camera_orientation

        # text1.set_text(str(camera_pose[0]))
        # text2.set_text(str(camera_pose[1]))
        # text3.set_text(str(camera_pose[2]))


        # max_unit_length = max(30, max(camera_pose[:3])) + 30

        # Decompose the camera coordinate
        arrow_length = - camera_pose[0] / camera_orientation[0]
        xs = [camera_pose[0], camera_pose[0] + camera_orientation[0] * arrow_length]
        ys = [camera_pose[1], camera_pose[1] + camera_orientation[1] * arrow_length]
        zs = [camera_pose[2], camera_pose[2] + camera_orientation[2] * arrow_length]


        # Plot camera location
        ax_camera = ax.scatter([camera_pose[0]], [camera_pose[1]], [camera_pose[2]],color='blue',lw=1)
        # item.append(ax_camera)
        # item.append(ax_label)

        arrow = Arrow3D(xs, ys, zs, mutation_scale=5, lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(arrow)

        flag = 1

        # temp = [camera_poseX[0] for k in range(frames)]
        # y = [camera_poseX[k] - temp[k] for k in range(len(x))]
        # n = frames - i
        # y[i:] = [0 for k in range(n)]
        # line1.set_data(x, y)   

        # temp = [camera_poseY[0] for k in range(frames)]
        # y = [camera_poseY[k] - temp[k] for k in range(len(x))]
        # n = frames - i
        # y[i:] = [0 for k in range(n)]
        # line2.set_data(x, y)   

        # temp = [camera_poseZ[0] for k in range(frames)]
        # y = [camera_poseZ[k] - temp[k] for k in range(len(x))]
        # n = frames - i
        # y[i:] = [0 for k in range(n)]
        # line3.set_data(x, y)   

        print i + frame_0

        writer.grab_frame()