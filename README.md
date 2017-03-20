# P4_LaneDetection
Project 4 - Lane Detection


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test1.jpg "Road Transformed"
[image3]: ./output_images/00_Orginal.jpg "Orginal Image"
[image4]: ./output_images/01_Orginal_Undist.jpg "Undistorted"
[image5]: ./output_images/02_Conversion2Binary.jpg "Converted to binary image"
[image6]: ./output_images/03_TransformationBinary.jpg "Transformed image"
[image7]: ./output_images/04_TransformationOrginal.jpg "Transformed orginal image"
[image8]: ./output_images/06_FindingLanesOverlay.jpg "Transformed orginal image with lane detection"
[image9]: ./output_images/07_FinalLanesOverlay.jpg "Final output with lane detection"
[image10]: ./output_images/Combimded2_HLS_Dir_vs_mag_Dir.jpg "Combind HLS & Direction vs. Magnidute & Direction"
[image11]: ./output_images/Combimded3_final.jpg "Binary detection"
[image12]: ./output_images/Combimded_X1_Y0.jpg "Binary detection gradients x==1 & gradients y ==0 "
[image13]: ./output_images/hls_binary.jpg "HLS binary detection"
[image14]: ./output_images/ThresholdedDirection.jpg "Direction binary detection"
[image15]: ./output_images/ThresholdedGradientx.jpg "Gradient x binary detection"
[image16]: ./output_images/ThresholdedGradienty.jpg "Gradient y binary detection"
[image17]: ./output_images/ThresholdedMagnidute.jpg "Thresholded Magnidute binary detection"
[image18]: ./output_images/hls_binary.jpg "HLS binary detection"
[image19]: ./output_images/hls_binary.jpg "HLS binary detection"


[video1]: ./output_images/output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The camera calibration is done within the module CAM_calibration.py.
All chessboard images in the folder camera_cal are readin as JPG.

Every image is process via the method calibrateOnFrame(img).
- First there is the conversion to a grayscale img
- Chessboard coners are detected by using the OpenCV2 methods (findChessboardCorners)
- The object and image points for all images are stored in two arrays
- After processing all images the camera matrix are calculated via the opencv method cv2.calibrateCamera
- Thoese values are stored as dictionary in a pickle file to reload on the acutal LineFinding process.

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
For isolating and converting the lines to a binary image of the road I used a combination of

- Gradient detection X axis versus Y axis  
![alt text][image15]
![alt text][image16]
- Gradient detection combinded (x==1 and y==0) results in:  
![alt text][image12]

- Addtional there is the HLS binary detection to get independ of lane colors
![alt text][image13]
- Also used is the direction binary image
![alt text][image14]
- With finally the method of magnidute
![alt text][image17]

- Combining thoses methods like ((hls_binary == 1) & (dir_binary == 1)) | ((mag_binary==1)& (dir_binary == 1)) comes up with
![alt text][image10]

All methods combinded together is a powerfull tool of creating a binary image with the key elements

![alt text][image11]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

The code of transforming the image is definded in the helper class ImageProcessing::transformImg.
Input to the function is the binary image with was created before in the pipeline.
A source and destination point is definded by:

```
 (h, w) = (img.shape[0], img.shape[1])
    source = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

```

Those values come up from "playing" around and checking other comments on the project channel. That is a important step and for real application there is a high need of correct tuning those parameters according to situations and use cases.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 564, 450      | 100, 0        | 
| 716, 450      | 1180,       |
| -100, 720     | 100, 720      |
| 1380, 720      | 1180, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detected the lanes I reused part of the code with was provided in the classroom.
So basically the binary pictured is sliced in 9 windows. Thoses windows were first located on the bottom of the picture where out of a histgrams result is a expected line. In every step there is a new window created to find the line in a range around the last windoww

![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The class Lane provides a object which all information about the detected lane. So out of the findLanes image processing a measurement is added to the line class.

...

    left_fitx,right_fitx,left_fit,right_fit,ploty,frame4 = ImageProcessing.findLines(frame3)
    
    LeftLine.addMeasurement(left_fitx,ploty,left_fit)
    RightLine.addMeasurement(right_fitx,ploty,right_fit)
...

On adding values to a line object there is running the filtering and measurement of curvature and position.
Out of the classroom there is used the 2. derivate to caluculate the  curvature:

...
    def CalcCurvature(self):
            # Define conversions in x and y from pixels space to meters
            ym_per_pix = (30/720) # meters per pixel in y dimension
            xm_per_pix = (3.7/700) # meters per pixel in x dimension

            ploty = self.ally
            fit_cr=self.current_fit
            y_eval = np.max(ploty)

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty*ym_per_pix, self.bestx*xm_per_pix, 2)
            # Calculate the new radii of curvature
            curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])


            return curverad
 ...

The raw measurments of the image processing with the position of the lane is filterd for the last 10 images.

s
####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used the opencv function warpPerspective with the same values as before - just the otherway around

...
 def transform2RealImg(undist,frame_warped,Minv,left_fitx,right_fitx,ploty):
     # Create an image to draw the lines on
     warp_zero = np.zeros_like(frame_warped).astype(np.uint8)
     color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

     # Recast the x and y points into usable format for cv2.fillPoly()
     pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
     pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
     pts = np.hstack((pts_left, pts_right))

     # Draw the lane onto the warped blank image
     cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

     # Warp the blank back to original image space using inverse perspective matrix (Minv)
     newwarp = cv2.warpPerspective(color_warp, Minv, (frame_warped.shape[1], frame_warped.shape[0])) 
     # Combine the result with the original image
     result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
     #plt.imshow(result)
     #plt.show()
     return result
     ...


![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/output.avi)

[video1]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  

There is strong need of tuning and changed threshold for binary image detection.
In various situation there are other lines detected than the lane lines.

Additional to binary image lane detection there shall be a region filtering adpative to vehicle speed, lateral acceleration and yaw rate.


