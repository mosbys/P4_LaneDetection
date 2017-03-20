import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
from random import randint

import ImageProcessing
import Line

iDebug=3;



def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ksize=15
    
    dir_binary = ImageProcessing.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.1))       
    grad_binary = ImageProcessing.abs_sobel_thresh(gray, orient='x', thresh_min=70, thresh_max=100)
    grad_binary2 = ImageProcessing.abs_sobel_thresh(gray, orient='y', thresh_min=50, thresh_max=100)

    mag_binary = ImageProcessing.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(50, 100))
    hls_binary = ImageProcessing.hls_select(img, thresh=(90, 254))

    gray[:,:]=0;
    
    gray[((grad_binary == 1) & (grad_binary2 == 0)) | ((hls_binary == 1) & (dir_binary == 1)) | ((mag_binary==1)& (dir_binary == 1))] = 254
    
    if (iDebug==3):
        combined1 = np.zeros_like(dir_binary).astype('uint8')
        combined2 = np.zeros_like(dir_binary).astype('uint8')
        combined1[((grad_binary == 1) & (grad_binary2 == 0))]=254
        combined2[((hls_binary == 1) & (dir_binary == 1)) | ((mag_binary==1)& (dir_binary == 1))]=254
    
    if (iDebugPlot==2):
        # save images
        #Oringal Img
        plt.imsave('OrginalImg.jpg',img)

        #Thresholded Gradient x'
        
        plt.imsave('ThresholdedGradientx.jpg',grad_binary,cmap='gray')

        #Thresholded Gradienty
        
        plt.imsave('ThresholdedGradienty.jpg',grad_binary2,cmap='gray')

        #Thresholded Direction
        
        plt.imsave('ThresholdedDirection.jpg',dir_binary,cmap='gray')

        #Thresholded Magnitude
        
        plt.imsave('ThresholdedMagnidute.jpg',mag_binary,cmap='gray')
        
        #Thresholded HSLS
        
        plt.imsave('hls_binary.jpg',hls_binary,cmap='gray')

        #Combinded X =1 Y=0 Gradient
        
        plt.imsave('Combimded_X1_Y0.jpg',combined1,cmap='gray')

        #Combinded((hls_binary == 1) & (dir_binary == 1)) | ((mag_binary==1)& (dir_binary == 1))
        
        plt.imsave('Combimded2_HLS_Dir_vs_mag_Dir.jpg',combined2,cmap='gray')

        #Combinded 1 +2
        
        plt.imsave('Combimded3_final.jpg',gray,cmap='gray')
        

        
            

    if (iDebug==4):
        # Plot the result
        f, ax = plt.subplots(4, 2, figsize=(24, 9))
        f.tight_layout()
        ax[0,0].imshow(img,cmap='gray')
        ax[0,0].set_title('Orignal Img', fontsize=25)
        ax[1,0].imshow(grad_binary, cmap='gray')
        ax[1,0].set_title('Thresholded Gradient x', fontsize=25)
        ax[0,1].imshow(mag_binary,cmap='gray')
        ax[0,1].set_title('Magnitude ', fontsize=25)
        ax[1,1].imshow(dir_binary, cmap='gray')
        ax[1,1].set_title('Direction of gradient x', fontsize=25)
        ax[2,0].imshow(hls_binary, cmap='gray')
        ax[2,0].set_title('HLS ', fontsize=25)
        ax[2,1].imshow(gray, cmap='gray')
        ax[2,1].set_title('Combinded Methods gradient', fontsize=25)
        ax[3,0].imshow(combined1,cmap='gray')
        ax[3,0].set_title('combined1', fontsize=25)
        ax[3,1].imshow(combined2, cmap='gray')
        ax[3,1].set_title('combined2', fontsize=25)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return gray


def processImg(frame,mtx,dist):
    
    
    frame1 = ImageProcessing.undistoredFrame(frame,mtx,dist)
    frame2 = pipeline(frame1)
    frame22  = cv2.cvtColor(frame2,cv2.COLOR_GRAY2BGR)
    outGradient.write(frame22)
    frame3= ImageProcessing.transformImg(frame2)
    frame33= ImageProcessing.transformImg(frame1)
    left_fitx,right_fitx,left_fit,right_fit,ploty,frame4 = ImageProcessing.findLines(frame3)
    
    LeftLine.addMeasurement(left_fitx,ploty,left_fit)
    RightLine.addMeasurement(right_fitx,ploty,right_fit)
    
    frame44=np.asarray(frame4,dtype='uint8')           # Debug out only
    frame5= cv2.addWeighted(frame44,0.4,frame33,0.6,0) # Debug out only
    outTransform.write(frame5)

    (h, w) = (frame1.shape[0], frame1.shape[1])
    source = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    transform_matrix = cv2.getPerspectiveTransform(destination,source)
    
    frame6 = ImageProcessing.transform2RealImg(frame,frame3,transform_matrix,LeftLine.bestx,RightLine.bestx,LeftLine.ally)
    
    if iDebugPlot==2:
        plt.imsave('00_Orginal.jpg',frame)
        plt.imsave('01_Orginal_Undist.jpg',frame1)
        plt.imsave('02_Conversion2Binary.jpg',frame2,cmap='gray')
        plt.imsave('03_TransformationBinary.jpg',frame3,cmap='gray')
        plt.imsave('04_TransformationOrginal.jpg',frame33)
        
        plt.imsave('05_FindingLanesOverlay.jpg',frame5,cmap='gray')
        plt.imsave('06_FinalLanesOverlay.jpg',frame6)
    return frame6


iDebugPlot=2
cap = cv2.VideoCapture(r'c:\users\Christoph\Documents\udacity\12_AdvLaneDetection\ProjectLaneFinding\ProjectLaneFinding\project_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))
outTransform = cv2.VideoWriter('outputTransform.avi', fourcc, 20.0, (1280,720))
outGradient = cv2.VideoWriter('outputGradient.avi', fourcc, 20.0, (1280,720))

with open('calibrate_camera.p', 'rb') as f:
	save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

iShowUndist=0
iFrame =1
LeftLine = Line.Line()
RightLine = Line.Line()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if (iDebug==2):
        frame1 = undistoredFrame(frame,mtx,dist)
    
    
        frame2 = pipeline(frame1)
        frame3= transformImg(frame2)
        frame33= transformImg(frame1)
             
        left_fitx,right_fitx,left_fit,right_fit,ploty,frame4 = findLines(frame3)
     
        left_curvature, right_curvature = CalcCurvature(left_fitx,right_fitx,left_fit,right_fit,ploty)
        # Display the resulting frame
     

    if iShowUndist==0:
        
        
        if (iDebug==1):
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame44=np.asarray(frame4,dtype='uint8')
            frame5= cv2.addWeighted(frame44,0.4,frame33,0.6,0)
            f, ax = plt.subplots(4, 2, figsize=(24, 9))
            f.tight_layout()

            ax[0,0].imshow(frame)
            ax[0,0].set_title('Orignal Img', fontsize=25)
            ax[1,0].imshow(frame1)
            ax[1,0].set_title('Undistorted Img', fontsize=25)
            ax[0,1].imshow(frame2,cmap='gray')
            ax[0,1].set_title('Pipeline ', fontsize=25)
            ax[1,1].imshow(frame3, cmap='gray')
            ax[1,1].set_title('Transform', fontsize=25)
            ax[2,0].imshow(frame4)
            ax[2,0].set_title('Find Lines ', fontsize=25)
            ax[2,1].imshow(frame44)
            ax[2,1].set_title('Find Lines unit8', fontsize=25)
            ax[3,0].imshow(frame33)
            ax[3,0].set_title('Frame 33 color', fontsize=25)
            ax[3,1].imshow(frame5)
            ax[3,1].set_title('Combinded 44 + 33', fontsize=25)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()


        frame5 = processImg(frame,mtx,dist)
        frame6 = cv2.cvtColor(frame5, cv2.COLOR_BGR2RGB)
        sTmp = 'Left Curvature: ' + str(LeftLine.radius_of_curvature) + ' Distance to left lane: ' + str(LeftLine.line_base_pos)
        cv2.putText(frame6,sTmp,(10,100), 0, 1,(255,255,255),1,cv2.LINE_AA)
        sTmp = 'Right Curvature Filter: ' + str(RightLine.radius_of_curvature) + ' Distance to right lane: ' + str(RightLine.line_base_pos)
        cv2.putText(frame6,sTmp,(10,150), 0, 1,(255,255,255),1,cv2.LINE_AA)
        #cv2.imshow('Out',frame6)
        out.write(frame6)
        iDebugPlot=0;
        if (iFrame%100==0):
            print("Current frame" + str(iFrame))
        if (iFrame>1000):
            break
        iFrame= iFrame+1
        #cv2.imshow('Org',frame)
        
        
    else:
        cv2.imshow('Out',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

       
        break
    elif cv2.waitKey(1) & 0xFF == ord('d'):
        iShowUndist=0
    elif cv2.waitKey(1) & 0xFF == ord('u'):
        iShowUndist=1
    
# When everything done, release the capture
cap.release()
out.release()
outTransform.release()
outGradient.release()
cv2.destroyAllWindows()
print("Current frame" + str(iFrame))



