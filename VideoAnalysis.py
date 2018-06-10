import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


def drawlines(img1, pts1):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img1, tuple(pt1), 2, color, -1)

    return img1


def findCenter(img, pts, color):
    if len(pts) > 0:
        ctX = np.sum(pts[:, 0]) / len(pts)
        ctY = np.sum(pts[:, 1]) / len(pts)
        cv2.circle(img, (int(ctX), int(ctY)), 2, color, -1)
    else:
        print("Points less than 2")
    return ctX, ctY


frame_i = 3267

filename = "/Users/yujieli/Documents/CFS_Video_Analysis-master/Results/EQ5T_OF.txt"

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=10,  # 10
                      blockSize=10)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(30, 30),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

color = np.random.randint(0, 255, (100, 3))

# resize = [260:r - 440,140:c - 920]

old_frame = cv2.imread("/Users/yujieli/Documents/CFS_Video_Analysis-master/Frames/frame3288.png")
r, c, a = old_frame.shape
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
# old_frame = old_frame[200:r - 60,50:c - 150]
old_frame = old_frame[800:r - 500, 1000:c - 1000]  # EQ5
# old_frame = old_frame[200:r - 100,0:c - 600]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

X = np.random.rand(100, 1000)
xs = p0[:, 0, 0]
ys = p0[:, 0, 1]

fig, ax = plt.subplots()
ax.set_title('click on point to plot time series')
line, = ax.plot(xs, ys, 'o', picker=5, lw=1)  # 5 points tolerance
centerX, centerY = 3180 / 2, 2160 / 2

line, = ax.plot(1094, 900, 'o', picker=5, lw=1, color=(0.2, 0.4, 0.8))  # 5 points tolerance
NUM = 4
N = 0
p00 = p0[:NUM, :, :]


def onpick(event):
    global N
    global p00

    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', ind, points)
    p00[N, :, :] = p0[ind[0], :, :]
    N = N + 1
    if N > NUM - 1:
        print("enough pts selected", p00)
        plt.close('all')
        return


def onClick(event):
    global N
    global p00
    if N > NUM - 1:
        print("enough pts selected", p00)
        plt.close('all')
        return
    x = int(event.xdata)
    y = int(event.ydata)
    print (x, y)
    p00[N, 0, 0] = x
    p00[N, 0, 1] = y
    N = N + 1


def on_key(event):
    global N
    if event.key is 'r':
        print('reset')
        N = 0
    if event.key is 'u':
        # p00[0,0,:] = 834, 596 #r
        # p00[1,0,:] = 1374, 103 #g
        # p00[2,0,:] = 1374, 873 #b
        p00[0, 0, :] = 428, 680  # r
        p00[1, 0, :] = 432, 218  # g
        p00[2, 0, :] = 1057, 214  # b
        p00[3, 0, :] = 1057, 680
        N = 4


fig.canvas.mpl_connect('pick_event', onpick)
# fig.canvas.mpl_connect('button_press_event', onClick)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.imshow(old_frame)
plt.show()

# cv2.waitKey(0)

# plt.close('all')

p0 = p00

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

with open(filename, "w") as f:
    color = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0), (0, 0, 255)]
    while (1):
        frame_i = frame_i + 1
        print frame_i
        frame = cv2.imread('/Users/yujieli/Documents/CFS_Video_Analysis-master/Frames/frame%d.png' % frame_i)

        if frame_i is None:
            break

        r, c, a = frame.shape
        frame = frame[800:r - 500, 1000:c - 1000]
        # frame = frame[260:r - 130,140:c - 580] #140 620
        # frame = frame[200:r - 100,0:c - 600]
        # frame = frame[240:r - 140,150:c - 620]
        # frame = frame[230:r - 100,230:c - 520]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks

        f.write(str(frame_i) + "\t")

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            f.write(str(a) + "\t" + str(b) + "\t")
        #     cv2.line(mask, (a, b), (c, d), color[i], 3)
        #     cv2.circle(frame, (a, b), 5, color[i], -1)
        # img = cv2.add(frame, mask)
        # r,c,a = frame.shape
        # img = img[220:r -120,150:c - 200,:]
        # cv2.imwrite('/Users/yujieli/Documents/CFS_Video_Analysis-master/Results/EQ5T_OFs/frame%d.png' % frame_i, img)
        # video.write(img)

        # img = cv2.resize(img, None, fx=0.5, fy=0.5)

        # cv2.imshow('img', img)
        #
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        f.write("\n")

# cap.release()
# video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()
