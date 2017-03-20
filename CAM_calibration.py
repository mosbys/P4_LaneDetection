import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
import glob
from random import randint



iDebugPlot=3



def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    #print(img.shape[::-1])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(gray.shape[::-1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist,mtx, dist


def calibrateOnFrame(img):
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    img2 = cv2.drawChessboardCorners(img, (9,6), corners, ret)


    objpoints = [] # 3D points in real world 
    imgpoints =[] # 2D points in image plane

    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    
    return corners, objp, ret


def undistoredFrame(img,mtx,dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

img_calibration = glob.glob(r'.\camera_cal\calibration*.jpg')


iIndex =0
objpoints = [] # 3D points in real world 
imgpoints =[] # 2D points in image plane
for imgPath in img_calibration:
    img = cv2.imread(imgPath)
    corners, objp ,ret= calibrateOnFrame(img)
    if (ret==True):
        imgpoints.append(corners)
        objpoints.append(objp)
    


img_undist, mtx, dist=cal_undistort(img,objpoints,imgpoints)

save_calibration = {'mtx': mtx, 'dist': dist}
with open('calibrate_camera.p', 'wb') as f:
    pickle.dump(save_calibration, f)


if (iDebugPlot==3):
    iRand=randint(0,len(img_calibration))
    img = cv2.imread(img_calibration[iRand])
    img_undist=undistoredFrame(img,mtx,dist)
    f, ax = plt.subplots(2, 1)
    f.tight_layout()
    ax[0].imshow(img)
    ax[0].set_title('Original Image ', fontsize=50)
    ax[1].imshow(img_undist)
    ax[1].set_title('Undistorted ', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
